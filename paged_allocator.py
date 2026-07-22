"""On-demand paged KV-cache allocator and its multi-seed stress suite."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from errors import BoundaryMissError, DoubleFreeError, OutOfMemoryError, UseAfterFreeError
from models import Request, SimulationResult, StepMetrics
from workload import clone_workload, generate_workload, group_by_arrival


class PhysicalBlockAllocator:
    """Own the global free-block pool, contents, and ownership metadata."""

    def __init__(self, total_blocks: int, block_size: int):
        """Create fixed-size physical blocks and mark all of them free."""
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a positive power of two")

        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = set(range(total_blocks))
        self.block_data: Dict[int, List[Optional[str]]] = {
            block_id: [None] * block_size for block_id in range(total_blocks)
        }
        self.owner: Dict[int, int] = {}

    @property
    def free_count(self) -> int:
        """Return the number of physical blocks available for allocation."""
        return len(self.free_blocks)

    @property
    def allocated_count(self) -> int:
        """Return the number of blocks currently owned by requests."""
        return self.total_blocks - len(self.free_blocks)

    def allocate(self, request_id: int) -> int:
        """Give one deterministic free block to a request and clear old data."""
        if not self.free_blocks:
            raise OutOfMemoryError(request_id, "no physical block is available")

        block_id = min(self.free_blocks)
        self.free_blocks.remove(block_id)
        self.block_data[block_id] = [None] * self.block_size
        self.owner[block_id] = request_id
        return block_id

    def free(self, block_id: int, request_id: int) -> None:
        """Return an allocated block while rejecting an accidental double-free."""
        if block_id not in self.block_data:
            raise DoubleFreeError(request_id, f"block {block_id} does not exist")
        if block_id in self.free_blocks:
            raise DoubleFreeError(request_id, f"block {block_id} is already free")

        self.free_blocks.add(block_id)
        self.owner.pop(block_id, None)
        self.block_data[block_id] = [None] * self.block_size

    def read(self, block_id: int, offset: int, request_id: int) -> Optional[str]:
        """Read one block offset after validating lifetime and boundaries."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError(request_id, f"read from freed block {block_id}")
        if not 0 <= offset < self.block_size:
            raise IndexError(f"offset {offset} outside block size {self.block_size}")
        return self.block_data[block_id][offset]

    def write(self, block_id: int, offset: int, value: str, request_id: int) -> None:
        """Write one simulated token after validating block lifetime and offset."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError(request_id, f"write to freed block {block_id}")
        if not 0 <= offset < self.block_size:
            raise IndexError(f"offset {offset} outside block size {self.block_size}")
        self.block_data[block_id][offset] = value

    def assert_consistent(self) -> None:
        """Prove free and owned blocks are disjoint and cover the whole pool."""
        assert self.free_blocks.isdisjoint(self.owner.keys())
        assert len(self.free_blocks) + len(self.owner) == self.total_blocks


@dataclass
class PageTable:
    """Translate a request's sequential logical blocks to physical block IDs."""

    entries: Dict[int, int] = field(default_factory=dict)

    def map(self, logical_block: int, physical_block: int) -> None:
        """Create one mapping while preventing accidental logical remapping."""
        if logical_block in self.entries:
            raise ValueError(f"logical block {logical_block} is already mapped")
        self.entries[logical_block] = physical_block

    def get(self, logical_block: int, request_id: int) -> int:
        """Resolve a logical block or raise the controlled boundary-miss error."""
        if logical_block not in self.entries:
            raise BoundaryMissError(
                request_id,
                f"logical block {logical_block} was never allocated",
            )
        return self.entries[logical_block]

    def clear(self) -> None:
        """Remove all logical-to-physical mappings after blocks are released."""
        self.entries.clear()


class PagedAllocator:
    """Allocate KV-cache blocks only as each request crosses a block boundary."""

    def __init__(self, total_tokens: int, block_size: int):
        """Build the physical pool and initialize active-request statistics."""
        if total_tokens < block_size:
            raise ValueError("total_tokens must be at least one block")
        self.total_tokens = total_tokens
        self.block_size = block_size
        self.total_blocks = total_tokens // block_size
        self.physical = PhysicalBlockAllocator(self.total_blocks, block_size)
        self.active: Dict[int, Request] = {}
        self.rejected = 0
        self.completed = 0
        self.peak_blocks_used = 0

    def _ensure_page_table(self, request: Request) -> PageTable:
        """Expose the request's mapping dictionary through PageTable checks."""
        table = PageTable(request.page_table)
        return table

    def generate_one_token(self, request: Request) -> None:
        """Map, allocate if necessary, and store the request's next token."""
        token_index = request.tokens_generated
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size
        table = self._ensure_page_table(request)

        # New blocks must appear in logical order with no gaps.
        if logical_block >= len(request.page_table):
            if logical_block != len(request.page_table):
                raise BoundaryMissError(
                    request.request_id,
                    f"cannot skip from {len(request.page_table)} to {logical_block}",
                )
            physical_block = self.physical.allocate(request.request_id)
            table.map(logical_block, physical_block)

        physical_block = table.get(logical_block, request.request_id)
        token = f"r{request.request_id}:t{token_index}"
        self.physical.write(physical_block, offset, token, request.request_id)
        request.token_sequence.append(token)
        request.tokens_generated += 1
        self.peak_blocks_used = max(
            self.peak_blocks_used,
            self.physical.allocated_count,
        )

    def access_token(self, request: Request, token_index: int) -> Optional[str]:
        """Translate a token index and read it from the mapped physical block."""
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size
        table = self._ensure_page_table(request)
        physical_block = table.get(logical_block, request.request_id)
        return self.physical.read(physical_block, offset, request.request_id)

    def release_request(self, request: Request) -> None:
        """Free every physical block owned through one request's page table."""
        allocated = list(request.page_table.values())
        for block_id in allocated:
            self.physical.free(block_id, request.request_id)

        request.page_table.clear()
        assert all(block_id in self.physical.free_blocks for block_id in allocated)

    def _internal_fragmentation(self) -> float:
        """Measure unused positions inside blocks held by active requests."""
        allocated_capacity = sum(len(req.page_table) * self.block_size for req in self.active.values())
        if allocated_capacity == 0:
            return 0.0
        used = sum(req.tokens_generated for req in self.active.values())
        return 100.0 * (allocated_capacity - used) / allocated_capacity

    def render_memory_map(self, width: int = 96) -> str:
        """Compress block ownership into a presentation-friendly ASCII map."""
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        chars: List[str] = []
        chunk_size = max(1, math.ceil(self.total_blocks / width))

        for start in range(0, self.total_blocks, chunk_size):
            block_ids = range(start, min(start + chunk_size, self.total_blocks))
            owners = {
                self.physical.owner[block_id]
                for block_id in block_ids
                if block_id not in self.physical.free_blocks
            }
            occupied = [
                block_id for block_id in block_ids if block_id not in self.physical.free_blocks
            ]
            if not occupied:
                chars.append(".")
            elif len(owners) == 1 and len(occupied) == len(list(block_ids)):
                chars.append(symbols[next(iter(owners)) % len(symbols)])
            else:
                chars.append("*")
        return "".join(chars)

    def assert_no_shared_blocks(self) -> None:
        """Prove active page tables do not share blocks and the pool is valid."""
        seen: set[int] = set()
        for request in self.active.values():
            blocks = set(request.page_table.values())
            assert len(blocks) == len(request.page_table)
            assert seen.isdisjoint(blocks), (
                f"request {request.request_id} shares a physical block"
            )
            seen.update(blocks)
        self.physical.assert_consistent()

    def run(
        self,
        requests: List[Request],
        print_maps: bool = True,
        map_width: int = 96,
        record_metrics: bool = True,
    ) -> SimulationResult:
        """Simulate arrivals and on-demand generation until the system drains."""
        arrivals = group_by_arrival(requests)
        final_arrival = max((req.arrival_time for req in requests), default=0)
        time_step = 0
        metrics: List[StepMetrics] = []
        internal_sum = 0.0
        metric_count = 0

        while time_step <= final_arrival or self.active:
            for request in arrivals.get(time_step, []):
                request.status = "active"
                self.active[request.request_id] = request

            finished: List[Request] = []
            rejected_now: List[Request] = []

            for request_id in sorted(list(self.active)):
                request = self.active.get(request_id)
                if request is None:
                    continue
                try:
                    self.generate_one_token(request)
                except OutOfMemoryError:
                    request.status = "rejected"
                    rejected_now.append(request)
                    self.rejected += 1
                    continue

                if request.tokens_generated >= request.actual_length:
                    request.status = "completed"
                    request.finished_at = time_step
                    finished.append(request)

            for request in rejected_now:
                self.release_request(request)
                self.active.pop(request.request_id, None)

            for request in finished:
                self.release_request(request)
                self.active.pop(request.request_id, None)
                self.completed += 1

            self.assert_no_shared_blocks()
            internal = self._internal_fragmentation()
            internal_sum += internal
            metric_count += 1
            memory_map = self.render_memory_map(map_width)

            if record_metrics:
                metrics.append(
                    StepMetrics(
                        time_step=time_step,
                        internal_fragmentation_percent=internal,
                        external_fragmentation_percent=0.0,
                        used_units=self.physical.allocated_count,
                        active_requests=len(self.active),
                        rejected_requests=self.rejected,
                        memory_map=memory_map,
                    )
                )

            if print_maps:
                print(
                    f"t={time_step:04d} [{memory_map}] "
                    f"active={len(self.active):3d} rejected={self.rejected:3d} "
                    f"internal={internal:6.2f}% external={0.0:6.2f}%"
                )

            time_step += 1

        assert self.physical.free_count == self.physical.total_blocks, "physical block leak"
        assert not self.physical.owner, "orphaned block owners remain"

        return SimulationResult(
            allocator_name="Paged",
            peak_blocks_used=self.peak_blocks_used,
            rejected_requests=self.rejected,
            internal_fragmentation_percent=(internal_sum / metric_count if metric_count else 0.0),
            external_fragmentation_percent=0.0,
            total_steps=time_step,
            completed_requests=self.completed,
            step_metrics=metrics,
        )


def run_stress_suite(
    requests_per_seed: int = 10_000,
    seeds: Iterable[int] = (11, 22, 33, 44, 55),
    block_sizes: Iterable[int] = (4, 8, 16, 32),
) -> Dict[int, Dict[str, int]]:
    """Run the required leak/boundary stress suite.

    The same five workloads are exercised at every required block size. The
    returned values are small summaries; any invariant failure raises.
    """
    report: Dict[int, Dict[str, int]] = {}
    seed_list = tuple(seeds)
    block_size_list = tuple(block_sizes)

    for block_size in block_size_list:
        completed = 0
        rejected = 0
        for seed in seed_list:
            workload = generate_workload(
                count=requests_per_seed,
                seed=seed,
                arrival_rate=4.0,
                mean_service_time=24.0,
                max_length=128,
            )
            simulator = PagedAllocator(total_tokens=4096, block_size=block_size)
            result = simulator.run(
                clone_workload(workload),
                print_maps=False,
                record_metrics=False,
            )
            completed += result.completed_requests
            rejected += result.rejected_requests

        report[block_size] = {
            "requests": requests_per_seed * len(seed_list),
            "completed": completed,
            "rejected": rejected,
            "assertion_failures": 0,
        }

    return report
