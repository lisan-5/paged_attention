"""Focused regression tests for paging, correctness, deadlock, and Copy-on-Write."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bonus import CopyOnWriteManager
from correctness import CorrectnessHarness
from deadlock import run_deadlock_demo
from models import Request
from paged_allocator import PagedAllocator, PhysicalBlockAllocator
from workload import clone_workload, generate_workload


class PagedAllocatorTests(unittest.TestCase):
    """Verify core paging boundaries, supported sizes, and error detection."""

    def test_boundary_tokens(self):
        """Check token-to-block mapping at offsets 0, 3, 4, and 8."""
        simulator = PagedAllocator(total_tokens=64, block_size=4)
        request = Request(1, 0, 16, 9, 9)
        for _ in range(9):
            simulator.generate_one_token(request)

        self.assertEqual(set(request.page_table), {0, 1, 2})
        self.assertEqual(simulator.access_token(request, 0), "r1:t0")
        self.assertEqual(simulator.access_token(request, 3), "r1:t3")
        self.assertEqual(simulator.access_token(request, 4), "r1:t4")
        self.assertEqual(simulator.access_token(request, 8), "r1:t8")

        simulator.release_request(request)
        self.assertEqual(simulator.physical.free_count, simulator.physical.total_blocks)

    def test_required_block_sizes(self):
        """Run the paged allocator with every block size named in the rubric."""
        for block_size in (4, 8, 16, 32):
            workload = generate_workload(200, seed=block_size, max_length=96)
            result = PagedAllocator(1024, block_size).run(
                clone_workload(workload), print_maps=False
            )
            self.assertEqual(
                result.completed_requests + result.rejected_requests,
                len(workload),
            )

    def test_failure_injections(self):
        """Require all three controlled allocator failures to be caught."""
        results = CorrectnessHarness().run_failure_injections()
        self.assertTrue(all(results.values()))


class BonusTests(unittest.TestCase):
    """Verify Copy-on-Write isolation and deadlock recovery."""

    def test_copy_on_write_isolation(self):
        """Prove that two completions can diverge without corrupting each other."""
        manager = CopyOnWriteManager(total_blocks=32, block_size=4)
        completions = manager.spawn(["a", "b", "c", "d", "e", "f"], 2)
        manager.write_token(completions[0], "left")
        manager.write_token(completions[1], "right")
        self.assertNotEqual(
            manager.visible_tokens(completions[0]),
            manager.visible_tokens(completions[1]),
        )
        manager.cancel(completions[0])
        manager.cancel(completions[1])
        self.assertEqual(len(manager.allocator.free_blocks), 32)

    def test_deadlock_recovery(self):
        """Confirm the crafted cycle is detected and broken in one step."""
        report = run_deadlock_demo(print_output=False)
        self.assertTrue(report.cycle_detected)
        self.assertTrue(report.recovered_same_step)
        self.assertEqual(report.victim, "A")


if __name__ == "__main__":
    unittest.main()
