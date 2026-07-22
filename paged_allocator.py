"""On-demand paged KV-cache allocation and multi-size stress testing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from errors import BoundaryMissError, DoubleFreeError, OutOfMemoryError, UseAfterFreeError
from models import Request, SimulationResult, StepMetrics
from workload import clone_workload, generate_workload, group_by_arrival


class PhysicalBlockAllocator:
    """Global pool of fixed-size physical blocks."""

    def __init__(self, total_blocks: int, block_size: int):
        """Create an empty physical block pool with validated dimensions."""
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        if block_size <= 0 or block_size & (block_size - 1):
            raise ValueError("block_size must be a power of two")

        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = set(range(total_blocks))
        self.data = {i: [None] * block_size for i in range(total_blocks)}
        self.owner: Dict[int, int] = {}

    @property
    def free_count(self) -> int:
        """Return the number of physical blocks currently available."""
        return len(self.free_blocks)

    @property
    def allocated_count(self) -> int:
        """Return the number of physical blocks currently in use."""
        return self.total_blocks - self.free_count

    def allocate(self, request_id: int) -> int:
        """Assign one free block to a request or raise an out-of-memory error."""
        if not self.free_blocks:
            raise OutOfMemoryError(request_id, "no free physical block")
        block_id = min(self.free_blocks)  # keeps seeded runs deterministic
        self.free_blocks.remove(block_id)
        self.data[block_id] = [None] * self.block_size
        self.owner[block_id] = request_id
        return block_id

    def free(self, block_id: int, request_id: int) -> None:
        """Release a block and detect attempts to free it more than once."""
        if block_id not in self.data or block_id in self.free_blocks:
            raise DoubleFreeError(request_id, f"block {block_id} is already free")
        self.free_blocks.add(block_id)
        self.owner.pop(block_id, None)
        self.data[block_id] = [None] * self.block_size

    def _check(self, block_id: int, offset: int, request_id: int) -> None:
        """Validate a block access and reject freed or invalid locations."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError(request_id, f"access to freed block {block_id}")
        if block_id not in self.data or offset not in range(self.block_size):
            raise IndexError(f"invalid block/offset {block_id}/{offset}")

    def read(self, block_id: int, offset: int, request_id: int) -> Optional[str]:
        """Read one token value after checking the block and offset."""
        self._check(block_id, offset, request_id)
        return self.data[block_id][offset]

    def write(self, block_id: int, offset: int, value: str, request_id: int) -> None:
        """Write one token value after checking the block and offset."""
        self._check(block_id, offset, request_id)
        self.data[block_id][offset] = value

    def assert_consistent(self) -> None:
        """Verify that every block is either free or owned, but never both."""
        assert self.free_blocks.isdisjoint(self.owner)
        assert self.free_count + len(self.owner) == self.total_blocks


@dataclass
class PageTable:
    """Map a request's logical block numbers to physical block IDs."""

    entries: Dict[int, int] = field(default_factory=dict)

    def add(self, logical: int, physical: int) -> None:
        """Add a new logical-to-physical mapping without overwriting one."""
        if logical in self.entries:
            raise ValueError(f"logical block {logical} is already mapped")
        self.entries[logical] = physical

    def get(self, logical: int, request_id: int) -> int:
        """Resolve a logical block or report that it was never allocated."""
        if logical not in self.entries:
            raise BoundaryMissError(request_id, f"logical block {logical} was not allocated")
        return self.entries[logical]


class PagedAllocator:
    """Allocate KV-cache blocks only when a request needs them."""

    def __init__(self, total_tokens: int, block_size: int):
        """Create the physical pool and initialize simulation counters."""
        if total_tokens < block_size:
            raise ValueError("total_tokens must fit at least one block")
        self.block_size = block_size
        self.total_blocks = total_tokens // block_size
        self.physical = PhysicalBlockAllocator(self.total_blocks, block_size)
        self.active: Dict[int, Request] = {}
        self.rejected = self.completed = self.peak_blocks_used = 0

    def generate_one_token(self, request: Request) -> None:
        """Allocate a page when needed and store the request's next token."""
        token_index = request.tokens_generated
        logical, offset = divmod(token_index, self.block_size)
        table = PageTable(request.page_table)

        if logical == len(table.entries):
            table.add(logical, self.physical.allocate(request.request_id))
        elif logical > len(table.entries):
            raise BoundaryMissError(request.request_id, "logical block allocation was skipped")

        block_id = table.get(logical, request.request_id)
        token = f"r{request.request_id}:t{token_index}"
        self.physical.write(block_id, offset, token, request.request_id)
        request.token_sequence.append(token)
        request.tokens_generated += 1
        self.peak_blocks_used = max(self.peak_blocks_used, self.physical.allocated_count)

    def access_token(self, request: Request, token_index: int) -> Optional[str]:
        """Translate a token index through the page table and read its value."""
        logical, offset = divmod(token_index, self.block_size)
        block_id = PageTable(request.page_table).get(logical, request.request_id)
        return self.physical.read(block_id, offset, request.request_id)

    def release_request(self, request: Request) -> None:
        """Free every physical block mapped to a finished request."""
        blocks = list(request.page_table.values())
        for block_id in blocks:
            self.physical.free(block_id, request.request_id)
        request.page_table.clear()
        assert all(block_id in self.physical.free_blocks for block_id in blocks)

    def _internal_fragmentation(self) -> float:
        """Measure unused token positions inside currently allocated blocks."""
        capacity = sum(len(r.page_table) * self.block_size for r in self.active.values())
        used = sum(r.tokens_generated for r in self.active.values())
        return 100 * (capacity - used) / capacity if capacity else 0.0

    def render_memory_map(self, width: int = 96) -> str:
        """Render block ownership as a compact presentation-friendly string."""
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        chunk_size = max(1, math.ceil(self.total_blocks / width))
        result = []

        for start in range(0, self.total_blocks, chunk_size):
            block_ids = list(range(start, min(start + chunk_size, self.total_blocks)))
            used = [i for i in block_ids if i not in self.physical.free_blocks]
            owners = {self.physical.owner[i] for i in used}
            if not used:
                result.append(".")
            elif len(used) == len(block_ids) and len(owners) == 1:
                result.append(symbols[next(iter(owners)) % len(symbols)])
            else:
                result.append("*")
        return "".join(result)

    def assert_no_shared_blocks(self) -> None:
        """Verify that active requests own distinct blocks and the pool is valid."""
        seen: set[int] = set()
        for request in self.active.values():
            blocks = set(request.page_table.values())
            assert len(blocks) == len(request.page_table)
            assert seen.isdisjoint(blocks), f"request {request.request_id} shares a block"
            seen.update(blocks)
        self.physical.assert_consistent()

    def run(
        self,
        requests: List[Request],
        print_maps: bool = True,
        map_width: int = 96,
        record_metrics: bool = True,
    ) -> SimulationResult:
        """Process arrivals and tokens until every request finishes or is rejected."""
        arrivals = group_by_arrival(requests)
        last_arrival = max((r.arrival_time for r in requests), default=0)
        metrics: List[StepMetrics] = []
        internal_total = 0.0
        step = 0

        while step <= last_arrival or self.active:
            for request in arrivals.get(step, []):
                request.status = "active"
                self.active[request.request_id] = request

            finished: List[Request] = []
            for request_id in sorted(self.active):
                request = self.active[request_id]
                try:
                    self.generate_one_token(request)
                except OutOfMemoryError:
                    request.status = "rejected"
                    self.rejected += 1
                    finished.append(request)
                    continue

                if request.tokens_generated >= request.actual_length:
                    request.status = "completed"
                    request.finished_at = step
                    self.completed += 1
                    finished.append(request)

            for request in finished:
                self.release_request(request)
                self.active.pop(request.request_id, None)

            self.assert_no_shared_blocks()
            internal = self._internal_fragmentation()
            internal_total += internal
            memory_map = self.render_memory_map(map_width)

            if record_metrics:
                metrics.append(StepMetrics(
                    step, internal, 0.0, self.physical.allocated_count,
                    len(self.active), self.rejected, memory_map
                ))
            if print_maps:
                print(
                    f"t={step:04d} [{memory_map}] active={len(self.active):3d} "
                    f"rejected={self.rejected:3d} internal={internal:6.2f}% external=  0.00%"
                )
            step += 1

        assert self.physical.free_count == self.total_blocks, "physical block leak"
        assert not self.physical.owner, "orphaned block owners remain"
        return SimulationResult(
            "Paged", self.peak_blocks_used, self.rejected,
            internal_total / step if step else 0.0, 0.0,
            step, self.completed, metrics
        )


def run_stress_suite(
    requests_per_seed: int = 10_000,
    seeds: Iterable[int] = (11, 22, 33, 44, 55),
    block_sizes: Iterable[int] = (4, 8, 16, 32),
) -> Dict[int, Dict[str, int]]:
    """Stress-test required block sizes across repeatable generated workloads."""
    seeds = tuple(seeds)
    report = {}
    for block_size in block_sizes:
        completed = rejected = 0
        for seed in seeds:
            workload = generate_workload(
                requests_per_seed, seed, arrival_rate=4.0,
                mean_service_time=24.0, max_length=128
            )
            result = PagedAllocator(4096, block_size).run(
                clone_workload(workload), print_maps=False, record_metrics=False
            )
            completed += result.completed_requests
            rejected += result.rejected_requests
        report[block_size] = {
            "requests": requests_per_seed * len(seeds),
            "completed": completed,
            "rejected": rejected,
            "assertion_failures": 0,
        }
    return report
