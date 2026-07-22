"""Preemption, deterministic correctness checks, and SLA-aware scheduling."""

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
    """Generate output from stable inputs so interrupted runs are comparable."""
    return f"request-{request_id}-token-{token_index}"


def sequence_hash(tokens: List[str]) -> str:
    """Create an unambiguous SHA-256 digest of an ordered token sequence."""
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def baseline_hash(request: Request) -> str:
    """Hash the uninterrupted output expected for one complete request."""
    tokens = [
        deterministic_token(request.request_id, index)
        for index in range(request.actual_length)
    ]
    return sequence_hash(tokens)


@dataclass
class PreemptionReport:
    """Summarize completion, eviction modes, correctness, and memory cleanup."""

    requests: int
    completed: int
    recomputations: int
    swaps: int
    hash_mismatches: int
    total_steps: int
    oversubscription_ratio: float
    orphaned_blocks: int


class PreemptionEngine:
    """Evict requests under pressure using recomputation or swap by threshold."""

    def __init__(
        self,
        total_blocks: int = 48,
        block_size: int = 4,
        threshold: int = 12,
        active_limit: int = 12,
    ):
        """Configure physical capacity, threshold T, and concurrent active slots."""
        self.physical = PhysicalBlockAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.threshold = threshold
        self.active_limit = active_limit

        self.active: Dict[int, Request] = {}
        self.waiting: Deque[Request] = deque()
        self.swap_store: Dict[int, List[str]] = {}
        self.recomputations = 0
        self.swaps = 0
        self.completed_hashes: Dict[int, str] = {}

    def _release_blocks(self, request: Request) -> None:
        """Free every GPU-style block mapped by an evicted or finished request."""
        for block_id in list(request.page_table.values()):
            self.physical.free(block_id, request.request_id)
        request.page_table.clear()

    def _select_victim(self, exclude_id: Optional[int] = None) -> Optional[Request]:
        """Choose the least-progressed request to minimize discarded work."""
        candidates = [
            request
            for request in self.active.values()
            if request.request_id != exclude_id
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda request: (request.tokens_generated, request.request_id))

    def _preempt(self, request: Request) -> None:
        """Free a victim and apply recompute below T or swap at/above T."""
        self._release_blocks(request)
        self.active.pop(request.request_id, None)

        if request.tokens_generated < self.threshold:
            request.tokens_generated = 0
            request.token_sequence.clear()
            request.preemption_mode = "recompute"
            self.swap_store.pop(request.request_id, None)
            self.recomputations += 1
        else:
            self.swap_store[request.request_id] = list(request.token_sequence)
            request.preemption_mode = "swap"
            self.swaps += 1

        request.status = "waiting"
        self.waiting.append(request)

    def _make_room(self, blocks_needed: int, exclude_id: Optional[int] = None) -> bool:
        """Preempt victims until enough blocks exist for the requesting operation."""
        while self.physical.free_count < blocks_needed:
            victim = self._select_victim(exclude_id)
            if victim is None:
                return False
            self._preempt(victim)
        return True

    def _restore(self, request: Request) -> bool:
        """Rebuild a swapped request's blocks and token data before resumption."""
        if request.tokens_generated == 0:
            return True

        saved = self.swap_store.get(request.request_id, request.token_sequence)
        blocks_needed = math.ceil(request.tokens_generated / self.block_size)
        if not self._make_room(blocks_needed, exclude_id=request.request_id):
            return False

        request.page_table.clear()
        for logical_block in range(blocks_needed):
            block_id = self.physical.allocate(request.request_id)
            request.page_table[logical_block] = block_id

        for token_index, token in enumerate(saved):
            logical_block = token_index // self.block_size
            offset = token_index % self.block_size
            block_id = request.page_table[logical_block]
            self.physical.write(block_id, offset, token, request.request_id)

        request.token_sequence = list(saved)
        return True

    def _admit_waiting(self) -> None:
        """Continuously fill open active slots with restorable waiting requests."""
        attempts = len(self.waiting)
        while self.waiting and len(self.active) < self.active_limit and attempts > 0:
            attempts -= 1
            request = self.waiting.popleft()
            if request.status in {"completed", "terminated"}:
                continue

            if not self._restore(request):
                self.waiting.appendleft(request)
                break

            request.status = "active"
            self.active[request.request_id] = request

    def _generate_one(self, request: Request) -> bool:
        """Generate one deterministic token, preempting another request if needed."""
        token_index = request.tokens_generated
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size

        if logical_block >= len(request.page_table):
            if not self._make_room(1, exclude_id=request.request_id):
                # This only occurs if the requester is the sole active request.
                self._preempt(request)
                return False
            block_id = self.physical.allocate(request.request_id)
            request.page_table[logical_block] = block_id

        block_id = request.page_table[logical_block]
        token = deterministic_token(request.request_id, token_index)
        self.physical.write(block_id, offset, token, request.request_id)
        request.token_sequence.append(token)
        request.tokens_generated += 1
        return True

    def run(self, requests: List[Request], max_steps: int = 200_000) -> PreemptionReport:
        """Run the oversubscribed workload and verify every final output hash."""
        arrivals = group_by_arrival(requests)
        final_arrival = max((request.arrival_time for request in requests), default=0)
        total_demand_blocks = sum(
            math.ceil(request.actual_length / self.block_size) for request in requests
        )
        oversubscription_ratio = total_demand_blocks / self.physical.total_blocks
        assert oversubscription_ratio >= 3.0, "workload is not oversubscribed by 3x"

        expected_hashes = {request.request_id: baseline_hash(request) for request in requests}
        time_step = 0
        completed = 0

        while (
            time_step <= final_arrival
            or self.waiting
            or self.active
        ):
            if time_step >= max_steps:
                raise RuntimeError("preemption simulation did not converge")

            for request in arrivals.get(time_step, []):
                request.status = "waiting"
                self.waiting.append(request)

            self._admit_waiting()

            for request_id in sorted(list(self.active)):
                request = self.active.get(request_id)
                if request is None:
                    continue
                generated = self._generate_one(request)
                if not generated or request.request_id not in self.active:
                    continue

                # Complete immediately so a later allocation in this same step
                # cannot choose an already-finished request as a victim.
                if request.tokens_generated >= request.actual_length:
                    self._release_blocks(request)
                    self.active.pop(request.request_id, None)
                    request.status = "completed"
                    request.finished_at = time_step
                    self.completed_hashes[request.request_id] = sequence_hash(request.token_sequence)
                    completed += 1

            self._admit_waiting()
            self.physical.assert_consistent()
            time_step += 1

        mismatches = sum(
            self.completed_hashes.get(request_id) != expected_hash
            for request_id, expected_hash in expected_hashes.items()
        )
        orphaned = self.physical.allocated_count
        assert completed == len(requests)
        assert mismatches == 0
        assert orphaned == 0

        return PreemptionReport(
            requests=len(requests),
            completed=completed,
            recomputations=self.recomputations,
            swaps=self.swaps,
            hash_mismatches=mismatches,
            total_steps=time_step,
            oversubscription_ratio=oversubscription_ratio,
            orphaned_blocks=orphaned,
        )


@dataclass
class SLAReport:
    """Summarize deadlines, budget stops, completions, and leaked blocks."""

    requests: int
    completed_normally: int
    deadline_misses: int
    budget_terminations: int
    orphaned_blocks: int
    total_steps: int


class SLAScheduler:
    """Admit requests by priority and deadline while enforcing token budgets."""

    def __init__(
        self,
        total_blocks: int = 512,
        block_size: int = 4,
        active_limit: int = 96,
    ):
        """Configure block capacity and initialize heap-based scheduler state."""
        self.physical = PhysicalBlockAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.active_limit = active_limit
        self.active: Dict[int, Request] = {}
        self.waiting_heap: List[Tuple[int, int, int, int]] = []
        self.requests: Dict[int, Request] = {}
        self.heap_order = 0
        self.deadline_missed_ids: Set[int] = set()
        self.completed_normally = 0
        self.budget_terminations = 0

    def _push(self, request: Request) -> None:
        """Queue a request by priority, absolute deadline, and stable arrival order."""
        request.status = "waiting"
        self.requests[request.request_id] = request
        heapq.heappush(
            self.waiting_heap,
            (
                request.priority,
                request.deadline_step,
                self.heap_order,
                request.request_id,
            ),
        )
        self.heap_order += 1

    def _admit(self, time_step: int) -> None:
        """Move the best-ranked waiting requests into available active slots."""
        while self.waiting_heap and len(self.active) < self.active_limit:
            _, _, _, request_id = heapq.heappop(self.waiting_heap)
            request = self.requests[request_id]
            if request.status != "waiting":
                continue
            request.status = "active"
            if request.first_started_at is None:
                request.first_started_at = time_step
            self.active[request_id] = request

    def _release(self, request: Request) -> None:
        """Free a completed or budget-terminated request without orphan blocks."""
        for block_id in list(request.page_table.values()):
            self.physical.free(block_id, request.request_id)
        request.page_table.clear()
        self.active.pop(request.request_id, None)

    def _generate_one(self, request: Request) -> bool:
        """Generate one token, returning false when physical capacity is full."""
        token_index = request.tokens_generated
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size

        if logical_block >= len(request.page_table):
            try:
                block_id = self.physical.allocate(request.request_id)
            except OutOfMemoryError:
                return False
            request.page_table[logical_block] = block_id

        block_id = request.page_table[logical_block]
        token = deterministic_token(request.request_id, token_index)
        self.physical.write(block_id, offset, token, request.request_id)
        request.token_sequence.append(token)
        request.tokens_generated += 1
        return True

    def _record_waiting_deadlines(self, time_step: int) -> None:
        """Track waiting duration and record each request that passes its deadline."""
        for _, _, _, request_id in self.waiting_heap:
            request = self.requests[request_id]
            if request.status != "waiting":
                continue
            request.wait_time = time_step - request.arrival_time
            if time_step > request.deadline_step:
                self.deadline_missed_ids.add(request_id)

    def run(self, requests: List[Request], max_steps: int = 20_000) -> SLAReport:
        """Schedule all arrivals with continuous admission and exact budget stops."""
        arrivals = group_by_arrival(requests)
        final_arrival = max((request.arrival_time for request in requests), default=0)
        time_step = 0
        finished_count = 0

        while time_step <= final_arrival or self.waiting_heap or self.active:
            if time_step >= max_steps:
                raise RuntimeError("SLA scheduler did not converge")

            for request in arrivals.get(time_step, []):
                self._push(request)

            self._admit(time_step)
            self._record_waiting_deadlines(time_step)

            finished: List[Tuple[Request, str]] = []
            for request_id in sorted(list(self.active)):
                request = self.active.get(request_id)
                if request is None:
                    continue

                if not self._generate_one(request):
                    # Capacity is genuinely unavailable; keep the request active
                    # and retry after another request releases blocks.
                    continue

                if request.tokens_generated >= request.actual_length:
                    finished.append((request, "completed"))
                elif (
                    request.token_budget is not None
                    and request.tokens_generated >= request.token_budget
                ):
                    finished.append((request, "budget"))

            for request, reason in finished:
                self._release(request)
                request.finished_at = time_step
                finished_count += 1
                if reason == "budget":
                    request.status = "terminated"
                    self.budget_terminations += 1
                    assert request.tokens_generated == request.token_budget
                else:
                    request.status = "completed"
                    self.completed_normally += 1

            # Continuous batching: fill newly opened slots immediately.
            self._admit(time_step)
            self._record_waiting_deadlines(time_step)
            self.physical.assert_consistent()
            time_step += 1

        orphaned = self.physical.allocated_count
        assert finished_count == len(requests)
        assert orphaned == 0

        return SLAReport(
            requests=len(requests),
            completed_normally=self.completed_normally,
            deadline_misses=len(self.deadline_missed_ids),
            budget_terminations=self.budget_terminations,
            orphaned_blocks=orphaned,
            total_steps=time_step,
        )


def run_required_preemption_demo(seed: int = 81) -> PreemptionReport:
    """Run the rubric's deterministic 500-request oversubscription scenario."""
    workload = generate_workload(
        count=500,
        seed=seed,
        arrival_rate=8.0,
        mean_service_time=35.0,
        max_length=128,
    )
    engine = PreemptionEngine(total_blocks=48, block_size=4, threshold=12, active_limit=12)
    return engine.run(clone_workload(workload))


def run_required_sla_demo(seed: int = 2025) -> SLAReport:
    """Run the rubric's 500-request priority, deadline, and budget scenario."""
    workload = generate_scheduler_workload(count=500, seed=seed)
    scheduler = SLAScheduler(total_blocks=512, block_size=4, active_limit=96)
    return scheduler.run(clone_workload(workload))
