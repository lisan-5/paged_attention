"""Controlled exceptions used to demonstrate allocator safety checks.

Keeping expected simulation failures separate from normal Python errors lets
the presentation harness report the failure type and request ID cleanly.
"""


class MemorySimulationError(Exception):
    """Base exception for controlled simulation failures."""

    failure_type = "MemorySimulationError"

    def __init__(self, request_id, message):
        """Attach the offending request ID to a human-readable error message."""
        self.request_id = request_id
        super().__init__(message)


class OutOfMemoryError(MemorySimulationError):
    """Raised when no free physical block is available."""

    failure_type = "OutOfMemoryError"


class DoubleFreeError(MemorySimulationError):
    """Raised when code attempts to return an already-free block."""

    failure_type = "DoubleFreeError"


class UseAfterFreeError(MemorySimulationError):
    """Raised when code accesses a block after its lifetime has ended."""

    failure_type = "UseAfterFreeError"


class BoundaryMissError(MemorySimulationError):
    """Raised when a logical token has no page-table mapping."""

    failure_type = "BoundaryMissError"
