"""Naive first-fit allocator used as the fragmentation comparison baseline."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from models import Request, SimulationResult, StepMetrics
from workload import group_by_arrival


class NaiveContiguousAllocator:
    """Reserve one maximum-length contiguous region for every active request."""

    def __init__(self, total_tokens: int, comparison_block_size: int = 16):
        """Create an empty token-slot array and initialize simulation counters."""
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        self.total_tokens = total_tokens
        self.comparison_block_size = comparison_block_size
        self.memory: List[Optional[int]] = [None] * total_tokens
        self.active: Dict[int, Request] = {}
        self.rejected = 0
        self.completed = 0
        self.peak_reserved_tokens = 0

    def _find_first_fit(self, length: int) -> Optional[int]:
        """Return the start of the first free run large enough for `length`."""
        run_start = 0
        run_length = 0

        for index, owner in enumerate(self.memory):
            if owner is None:
                if run_length == 0:
                    run_start = index
                run_length += 1
                if run_length >= length:
                    return run_start
            else:
                run_length = 0

        return None

    def _reserve(self, request: Request) -> bool:
        """Reserve `max_length` contiguous slots or report that no run fits."""
        start = self._find_first_fit(request.max_length)
        if start is None:
            return False

        end = start + request.max_length
        for index in range(start, end):
            self.memory[index] = request.request_id

        request.allocation_start = start
        request.allocation_size = request.max_length
        request.status = "active"
        self.active[request.request_id] = request
        self.peak_reserved_tokens = max(
            self.peak_reserved_tokens,
            sum(req.allocation_size for req in self.active.values()),
        )
        return True

    def _release(self, request: Request) -> None:
        """Return every slot reserved by a completed request to memory."""
        if request.allocation_start is None:
            return

        start = request.allocation_start
        end = start + request.allocation_size
        for index in range(start, end):
            if self.memory[index] == request.request_id:
                self.memory[index] = None

        request.allocation_start = None
        request.allocation_size = 0
        self.active.pop(request.request_id, None)

    def _free_gaps(self) -> List[int]:
        """Measure each contiguous run of unallocated token slots."""
        gaps: List[int] = []
        current = 0

        for owner in self.memory:
            if owner is None:
                current += 1
            elif current:
                gaps.append(current)
                current = 0

        if current:
            gaps.append(current)
        return gaps

    def _internal_fragmentation(self) -> float:
        """Measure reserved capacity not yet reached by generated tokens."""
        reserved = sum(req.allocation_size for req in self.active.values())
        if reserved == 0:
            return 0.0

        idle = sum(
            max(0, req.allocation_size - req.tokens_generated)
            for req in self.active.values()
        )
        return 100.0 * idle / reserved

    def _external_fragmentation_for(self, needed: Optional[int]) -> float:
        """Measure free slots trapped in gaps smaller than an arriving request."""
        if needed is None:
            return 0.0

        gaps = self._free_gaps()
        total_free = sum(gaps)
        if total_free == 0:
            return 0.0

        unusable = sum(gap for gap in gaps if gap < needed)
        return 100.0 * unusable / total_free

    def render_memory_map(self, width: int = 96) -> str:
        """Render a compact map; each character summarizes a memory chunk."""
        if width <= 0:
            return ""

        chunk_size = max(1, math.ceil(self.total_tokens / width))
        chars: List[str] = []
        symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        for start in range(0, self.total_tokens, chunk_size):
            chunk = self.memory[start : start + chunk_size]
            owners = {owner for owner in chunk if owner is not None}
            if not owners:
                chars.append(".")
            elif len(owners) == 1 and all(owner is not None for owner in chunk):
                owner = next(iter(owners))
                chars.append(symbols[owner % len(symbols)])
            else:
                chars.append("*")

        return "".join(chars)

    def run(
        self,
        requests: List[Request],
        print_maps: bool = True,
        map_width: int = 96,
    ) -> SimulationResult:
        """Advance arrivals and token generation until every request is resolved."""
        arrivals = group_by_arrival(requests)
        final_arrival = max((req.arrival_time for req in requests), default=0)
        time_step = 0
        step_metrics: List[StepMetrics] = []

        while time_step <= final_arrival or self.active:
            rejected_fragmentation_samples: List[float] = []

            for request in arrivals.get(time_step, []):
                request.status = "waiting"
                ext_before_admission = self._external_fragmentation_for(request.max_length)
                if not self._reserve(request):
                    request.status = "rejected"
                    self.rejected += 1
                    rejected_fragmentation_samples.append(ext_before_admission)

            finished: List[Request] = []
            for request_id in sorted(self.active):
                request = self.active[request_id]
                request.tokens_generated += 1
                if request.tokens_generated >= request.actual_length:
                    request.status = "completed"
                    request.finished_at = time_step
                    finished.append(request)

            for request in finished:
                self._release(request)
                self.completed += 1

            internal = self._internal_fragmentation()
            external = (
                sum(rejected_fragmentation_samples) / len(rejected_fragmentation_samples)
                if rejected_fragmentation_samples
                else 0.0
            )
            memory_map = self.render_memory_map(map_width)

            metric = StepMetrics(
                time_step=time_step,
                internal_fragmentation_percent=internal,
                external_fragmentation_percent=external,
                used_units=sum(owner is not None for owner in self.memory),
                active_requests=len(self.active),
                rejected_requests=self.rejected,
                memory_map=memory_map,
            )
            step_metrics.append(metric)

            if print_maps:
                print(
                    f"t={time_step:04d} [{memory_map}] "
                    f"active={len(self.active):3d} rejected={self.rejected:3d} "
                    f"internal={internal:6.2f}% external={external:6.2f}%"
                )

            time_step += 1

        average_internal = (
            sum(item.internal_fragmentation_percent for item in step_metrics) / len(step_metrics)
            if step_metrics
            else 0.0
        )
        average_external = (
            sum(item.external_fragmentation_percent for item in step_metrics) / len(step_metrics)
            if step_metrics
            else 0.0
        )

        return SimulationResult(
            allocator_name="Naive",
            peak_blocks_used=math.ceil(
                self.peak_reserved_tokens / self.comparison_block_size
            ),
            rejected_requests=self.rejected,
            internal_fragmentation_percent=average_internal,
            external_fragmentation_percent=average_external,
            total_steps=time_step,
            completed_requests=self.completed,
            step_metrics=step_metrics,
        )
