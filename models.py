
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Request:
    """Represent one inference request and all of its mutable runtime state."""

    request_id: int
    arrival_time: int
    max_length: int
    actual_length: int
    service_time: int

    priority: int = 2
    deadline: int = 30
    token_budget: Optional[int] = None

    status: str = "new"
    tokens_generated: int = 0
    page_table: Dict[int, int] = field(default_factory=dict)
    token_sequence: List[str] = field(default_factory=list)

    allocation_start: Optional[int] = None
    allocation_size: int = 0
    wait_time: int = 0
    first_started_at: Optional[int] = None
    finished_at: Optional[int] = None
    preemption_mode: Optional[str] = None

    @property
    def deadline_step(self) -> int:
        """Convert the relative waiting deadline into an absolute time step."""
        return self.arrival_time + self.deadline

    def reset_runtime(self) -> None:
        """Clear simulation state so a copied workload can be reused fairly."""
        self.status = "new"
        self.tokens_generated = 0
        self.page_table.clear()
        self.token_sequence.clear()
        self.allocation_start = None
        self.allocation_size = 0
        self.wait_time = 0
        self.first_started_at = None
        self.finished_at = None
        self.preemption_mode = None


@dataclass
class StepMetrics:
    """Capture allocator measurements for one simulation time step."""

    time_step: int
    internal_fragmentation_percent: float
    external_fragmentation_percent: float
    used_units: int
    active_requests: int
    rejected_requests: int
    memory_map: str = ""


@dataclass
class SimulationResult:
    """Summarize one complete allocator run for reporting and comparison."""

    allocator_name: str
    peak_blocks_used: int
    rejected_requests: int
    internal_fragmentation_percent: float
    external_fragmentation_percent: float
    total_steps: int
    completed_requests: int
    step_metrics: List[StepMetrics] = field(default_factory=list)
