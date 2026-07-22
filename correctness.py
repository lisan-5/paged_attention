"""Invariant checks and deliberate failure injections for allocator safety."""

from __future__ import annotations

from typing import Dict

from errors import BoundaryMissError, DoubleFreeError, MemorySimulationError, UseAfterFreeError
from models import Request
from paged_allocator import PagedAllocator, PhysicalBlockAllocator


class CorrectnessHarness:
    """Prove that blocks are isolated and controlled errors are detected."""

    def check_active_requests(self, active: Dict[int, Request]) -> None:
        """Ensures two active requests never own the same physical block."""
        seen: set[int] = set()
        for request in active.values():
            blocks = set(request.page_table.values())
            assert seen.isdisjoint(blocks), (
                f"physical block overlap for request {request.request_id}"
            )
            seen.update(blocks)

    @staticmethod
    def _print_caught(error: MemorySimulationError) -> None:
        """Print the exact failure type and request ID required by the task."""
        print(f"Caught {error.failure_type} for request {error.request_id}")

    def inject_double_free(self) -> bool:
        """Free one block twice and confirm the allocator rejects the second call."""
        allocator = PhysicalBlockAllocator(total_blocks=2, block_size=4)
        request_id = 9001
        block_id = allocator.allocate(request_id)
        allocator.free(block_id, request_id)
        try:
            allocator.free(block_id, request_id)
        except DoubleFreeError as error:
            self._print_caught(error)
            return True
        return False

    def inject_use_after_free(self) -> bool:
        """Read a released block and confirm the lifetime check catches it."""
        allocator = PhysicalBlockAllocator(total_blocks=2, block_size=4)
        request_id = 9002
        block_id = allocator.allocate(request_id)
        allocator.write(block_id, 0, "temporary", request_id)
        allocator.free(block_id, request_id)
        try:
            allocator.read(block_id, 0, request_id)
        except UseAfterFreeError as error:
            self._print_caught(error)
            return True
        return False

    def inject_boundary_miss(self) -> bool:
        """Access an unmapped token to verify page-table boundary protection."""
        simulator = PagedAllocator(total_tokens=32, block_size=4)
        request = Request(
            request_id=9003,
            arrival_time=0,
            max_length=8,
            actual_length=8,
            service_time=8,
        )
        try:
            simulator.access_token(request, token_index=4)
        except BoundaryMissError as error:
            self._print_caught(error)
            return True
        return False

    def run_failure_injections(self) -> Dict[str, bool]:
        """Run all required failure demonstrations and require every one to pass."""
        results = {
            "double_free": self.inject_double_free(),
            "use_after_free": self.inject_use_after_free(),
            "boundary_miss": self.inject_boundary_miss(),
        }
        assert all(results.values()), "one or more injected failures were not caught"
        return results
