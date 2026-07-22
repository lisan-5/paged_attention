"""Preemption and SLA schedulers for oversubscribed inference workloads."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

from errors import OutOfMemoryError
from models import Request
from paged_allocator import PhysicalBlockAllocator
from workload import clone_workload, generate_scheduler_workload, generate_workload, group_by_arrival


def deterministic_token(request_id: int, token_index: int) -> str:
    """Create repeatable token content for later correctness verification."""
    return f"request-{request_id}-token-{token_index}"


def sequence_hash(tokens: List[str]) -> str:
    """Hash a generated token sequence so restored output can be verified."""
    return hashlib.sha256("\0".join(tokens).encode()).hexdigest()


def expected_hash(request: Request) -> str:
    """Calculate the hash a correctly completed request should produce."""
    tokens = [deterministic_token(request.request_id, i) for i in range(request.actual_length)]
    return sequence_hash(tokens)


def release_blocks(physical: PhysicalBlockAllocator, request: Request) -> None:
    """Return all blocks owned by a request and clear its page table."""
    for block_id in list(request.page_table.values()):
        physical.free(block_id, request.request_id)
    request.page_table.clear()


@dataclass
class PreemptionReport:
    """Summarize preemption activity, correctness, and memory cleanup."""

    requests: int
    completed: int
    recomputations: int
    swaps: int
    hash_mismatches: int
    total_steps: int
    oversubscription_ratio: float
    orphaned_blocks: int


class PreemptionEngine:
    """Preempt short requests by recomputation and longer ones by swapping."""

    def __init__(
        self,
        total_blocks: int = 48,
        block_size: int = 4,
        threshold: int = 12,
        active_limit: int = 12,
    ):
        """Configure limited memory, admission capacity, and preemption policy."""
        self.physical = PhysicalBlockAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.threshold = threshold
        self.active_limit = active_limit
        self.active: Dict[int, Request] = {}
        self.waiting: Deque[Request] = deque()
        self.swap_store: Dict[int, List[str]] = {}
        self.recomputations = self.swaps = 0
        self.completed_hashes: Dict[int, str] = {}

    def _preempt(self, request: Request) -> None:
        """Pause a request using recomputation for short work or swap for long work."""
        release_blocks(self.physical, request)
        self.active.pop(request.request_id, None)

        if request.tokens_generated < self.threshold:
            request.tokens_generated = 0
            request.token_sequence.clear()
            request.preemption_mode = "recompute"
            self.swap_store.pop(request.request_id, None)
            self.recomputations += 1
        else:
            request.preemption_mode = "swap"
            self.swap_store[request.request_id] = list(request.token_sequence)
            self.swaps += 1

        request.status = "waiting"
        self.waiting.append(request)

    def _make_room(self, blocks_needed: int, exclude_id: Optional[int] = None) -> bool:
        """Preempt eligible requests until the required blocks are available."""
        while self.physical.free_count < blocks_needed:
            candidates = [r for r in self.active.values() if r.request_id != exclude_id]
            if not candidates:
                return False
            victim = min(candidates, key=lambda r: (r.tokens_generated, r.request_id))
            self._preempt(victim)
        return True

    def _restore(self, request: Request) -> bool:
        """Reallocate blocks and restore saved tokens before resuming a request."""
        if request.tokens_generated == 0:
            return True

        saved = self.swap_store.get(request.request_id, request.token_sequence)
        block_count = math.ceil(request.tokens_generated / self.block_size)
        if not self._make_room(block_count, request.request_id):
            return False

        for logical in range(block_count):
            request.page_table[logical] = self.physical.allocate(request.request_id)
        for index, token in enumerate(saved):
            logical, offset = divmod(index, self.block_size)
            self.physical.write(
                request.page_table[logical], offset, token, request.request_id
            )
        request.token_sequence = list(saved)
        return True

    def _admit(self) -> None:
        """Move waiting requests into open active slots when they can be restored."""
        attempts = len(self.waiting)
        while self.waiting and len(self.active) < self.active_limit and attempts:
            attempts -= 1
            request = self.waiting.popleft()
            if request.status in {"completed", "terminated"}:
                continue
            if not self._restore(request):
                self.waiting.appendleft(request)
                break
            request.status = "active"
            self.active[request.request_id] = request

    def _generate(self, request: Request) -> bool:
        """Generate one token, making room through preemption when necessary."""
        index = request.tokens_generated
        logical, offset = divmod(index, self.block_size)

        if logical == len(request.page_table):
            if not self._make_room(1, request.request_id):
                self._preempt(request)
                return False
            request.page_table[logical] = self.physical.allocate(request.request_id)

        token = deterministic_token(request.request_id, index)
        self.physical.write(
            request.page_table[logical], offset, token, request.request_id
        )
        request.token_sequence.append(token)
        request.tokens_generated += 1
        return True

    def run(self, requests: List[Request], max_steps: int = 200_000) -> PreemptionReport:
        """Run the oversubscribed workload and verify all final token hashes."""
        arrivals = group_by_arrival(requests)
        last_arrival = max((r.arrival_time for r in requests), default=0)
        demand = sum(math.ceil(r.actual_length / self.block_size) for r in requests)
        oversubscription = demand / self.physical.total_blocks
        assert oversubscription >= 3

        expected = {r.request_id: expected_hash(r) for r in requests}
        completed = step = 0

        while step <= last_arrival or self.waiting or self.active:
            if step >= max_steps:
                raise RuntimeError("preemption simulation did not converge")

            for request in arrivals.get(step, []):
                request.status = "waiting"
                self.waiting.append(request)
            self._admit()

            for request_id in sorted(list(self.active)):
                request = self.active.get(request_id)
                if request is None or not self._generate(request):
                    continue
                if request.request_id not in self.active:
                    continue
                if request.tokens_generated >= request.actual_length:
                    release_blocks(self.physical, request)
                    self.active.pop(request.request_id)
                    request.status = "completed"
                    request.finished_at = step
                    self.completed_hashes[request.request_id] = sequence_hash(request.token_sequence)
                    completed += 1

            self._admit()  # continuous batching
            self.physical.assert_consistent()
            step += 1

        mismatches = sum(
            self.completed_hashes.get(request_id) != value
            for request_id, value in expected.items()
        )
        orphaned = self.physical.allocated_count
        assert completed == len(requests) and mismatches == 0 and orphaned == 0
        return PreemptionReport(
            len(requests), completed, self.recomputations, self.swaps,
            mismatches, step, oversubscription, orphaned
        )


@dataclass
class SLAReport:
    """Summarize normal completions, deadline misses, and budget stops."""

    requests: int
    completed_normally: int
    deadline_misses: int
    budget_terminations: int
    orphaned_blocks: int
    total_steps: int


class SLAScheduler:
    """Serve by priority/deadline and stop exactly at the token budget."""

    def __init__(
        self,
        total_blocks: int = 512,
        block_size: int = 4,
        active_limit: int = 96,
    ):
        """Configure block capacity and priority-based admission state."""
        self.physical = PhysicalBlockAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.active_limit = active_limit
        self.active: Dict[int, Request] = {}
        self.waiting: List[Tuple[int, int, int, int]] = []
        self.requests: Dict[int, Request] = {}
        self.order = 0
        self.missed: Set[int] = set()
        self.normal = self.budget_stops = 0

    def _queue(self, request: Request) -> None:
        """Queue a request by priority, deadline, and stable arrival order."""
        request.status = "waiting"
        self.requests[request.request_id] = request
        heapq.heappush(
            self.waiting,
            (request.priority, request.deadline_step, self.order, request.request_id),
        )
        self.order += 1

    def _admit(self, step: int) -> None:
        """Admit the highest-ranked waiting requests into available slots."""
        while self.waiting and len(self.active) < self.active_limit:
            *_, request_id = heapq.heappop(self.waiting)
            request = self.requests[request_id]
            if request.status != "waiting":
                continue
            request.status = "active"
            request.first_started_at = request.first_started_at or step
            self.active[request_id] = request

    def _generate(self, request: Request) -> bool:
        """Generate one token, returning false when no block is available."""
        index = request.tokens_generated
        logical, offset = divmod(index, self.block_size)
        if logical == len(request.page_table):
            try:
                request.page_table[logical] = self.physical.allocate(request.request_id)
            except OutOfMemoryError:
                return False

        token = deterministic_token(request.request_id, index)
        self.physical.write(
            request.page_table[logical], offset, token, request.request_id
        )
        request.token_sequence.append(token)
        request.tokens_generated += 1
        return True

    def _check_deadlines(self, step: int) -> None:
        """Update wait times and record requests that pass their deadlines."""
        for *_, request_id in self.waiting:
            request = self.requests[request_id]
            if request.status == "waiting":
                request.wait_time = step - request.arrival_time
                if step > request.deadline_step:
                    self.missed.add(request_id)

    def run(self, requests: List[Request], max_steps: int = 20_000) -> SLAReport:
        """Schedule all requests while enforcing deadlines and token budgets."""
        arrivals = group_by_arrival(requests)
        last_arrival = max((r.arrival_time for r in requests), default=0)
        finished = step = 0

        while step <= last_arrival or self.waiting or self.active:
            if step >= max_steps:
                raise RuntimeError("SLA scheduler did not converge")

            for request in arrivals.get(step, []):
                self._queue(request)
            self._admit(step)
            self._check_deadlines(step)

            done: List[Tuple[Request, str]] = []
            for request_id in sorted(self.active):
                request = self.active[request_id]
                if not self._generate(request):
                    continue
                if request.tokens_generated >= request.actual_length:
                    done.append((request, "completed"))
                elif request.token_budget is not None and request.tokens_generated >= request.token_budget:
                    done.append((request, "budget"))

            for request, reason in done:
                release_blocks(self.physical, request)
                self.active.pop(request.request_id)
                request.finished_at = step
                finished += 1
                if reason == "budget":
                    request.status = "terminated"
                    self.budget_stops += 1
                    assert request.tokens_generated == request.token_budget
                else:
                    request.status = "completed"
                    self.normal += 1

            self._admit(step)
            self._check_deadlines(step)
            self.physical.assert_consistent()
            step += 1

        orphaned = self.physical.allocated_count
        assert finished == len(requests) and orphaned == 0
        return SLAReport(
            len(requests), self.normal, len(self.missed),
            self.budget_stops, orphaned, step
        )


def run_required_preemption_demo(seed: int = 81) -> PreemptionReport:
    """Run the required 500-request oversubscribed preemption scenario."""
    workload = generate_workload(
        500, seed, arrival_rate=8.0, mean_service_time=35.0, max_length=128
    )
    return PreemptionEngine().run(clone_workload(workload))


def run_required_sla_demo(seed: int = 2025) -> SLAReport:
    """Run the required priority, deadline, and token-budget scenario."""
    return SLAScheduler().run(
        clone_workload(generate_scheduler_workload(500, seed))
    )
