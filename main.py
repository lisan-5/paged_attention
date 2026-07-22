
from __future__ import annotations

import argparse

from bonus import run_cow_demo
from correctness import CorrectnessHarness
from deadlock import run_deadlock_demo
from naive_allocator import NaiveContiguousAllocator
from paged_allocator import PagedAllocator, run_stress_suite
from scheduler import run_required_preemption_demo, run_required_sla_demo
from workload import clone_workload, generate_workload


def print_comparison_table(naive, paged) -> None:
    """Print the two allocator results side by side using aligned columns."""
    headers = (
        "Allocator",
        "Peak blocks",
        "Rejected",
        "Internal frag.",
        "External frag.",
    )
    rows = [
        (
            naive.allocator_name,
            str(naive.peak_blocks_used),
            str(naive.rejected_requests),
            f"{naive.internal_fragmentation_percent:.2f}%",
            f"{naive.external_fragmentation_percent:.2f}%",
        ),
        (
            paged.allocator_name,
            str(paged.peak_blocks_used),
            str(paged.rejected_requests),
            f"{paged.internal_fragmentation_percent:.2f}%",
            f"{paged.external_fragmentation_percent:.2f}%",
        ),
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def format_row(row):
        """Pad one output row to the width calculated for each column."""
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))


def run_task_1(print_maps: bool, run_full_stress: bool) -> None:
    """Compare allocators, inject failures, and run block-size stress tests."""
    print("\n=== TASK 1: Naive vs PagedAllocator ===")
    workload = generate_workload(count=200, seed=42)

    if print_maps:
        print("\nNaive per-step memory map")
    naive = NaiveContiguousAllocator(
        total_tokens=4096,
        comparison_block_size=16,
    ).run(clone_workload(workload), print_maps=print_maps)

    if print_maps:
        print("\nPaged per-step memory map")
    paged = PagedAllocator(total_tokens=4096, block_size=16).run(
        clone_workload(workload),
        print_maps=print_maps,
    )

    print("\nHead-to-head comparison (same seed and same workload)")
    print_comparison_table(naive, paged)

    print("\nCorrectness harness")
    CorrectnessHarness().run_failure_injections()

    if run_full_stress:
        print("\n10,000 requests x 5 seeds for each required block size")
        stress = run_stress_suite()
    else:
        print("\nQuick stress mode")
        stress = run_stress_suite(requests_per_seed=500, seeds=(11, 22))

    for block_size, values in stress.items():
        print(
            f"block_size={block_size:2d} requests={values['requests']:6d} "
            f"completed={values['completed']:6d} rejected={values['rejected']:4d} "
            f"assertion_failures={values['assertion_failures']}"
        )


def run_task_2() -> None:
    """Demonstrate preemption, SLA scheduling, and deadlock recovery."""
    print("\n=== TASK 2: Preemption and Multi-Tenant Scheduling ===")
    preemption = run_required_preemption_demo()
    print(
        "Preemption: "
        f"completed={preemption.completed}/{preemption.requests}, "
        f"oversubscription={preemption.oversubscription_ratio:.2f}x, "
        f"recomputations={preemption.recomputations}, swaps={preemption.swaps}, "
        f"hash_mismatches={preemption.hash_mismatches}, "
        f"orphaned_blocks={preemption.orphaned_blocks}"
    )

    sla = run_required_sla_demo()
    print(
        "SLA scheduler: "
        f"requests={sla.requests}, deadline_misses={sla.deadline_misses}, "
        f"budget_terminations={sla.budget_terminations}, "
        f"normal_completions={sla.completed_normally}, "
        f"orphaned_blocks={sla.orphaned_blocks}"
    )

    deadlock = run_deadlock_demo(print_output=True)
    assert deadlock.recovered_same_step


def run_task_3() -> None:
    """Run the Copy-on-Write bonus demonstration."""
    print("\n=== TASK 3: Copy-on-Write Bonus ===")
    cow = run_cow_demo()
    print(
        "Copy-on-Write: "
        f"N={cow.tested_completion_counts}, "
        f"isolation_failures={cow.isolation_failures}, leaked_blocks={cow.leaked_blocks}"
    )


def main() -> None:
    """Parse command-line flags and execute the complete project in order."""
    parser = argparse.ArgumentParser(description="Pure Python PagedAttention simulation")
    parser.add_argument(
        "--maps",
        action="store_true",
        help="print the required per-step memory maps",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a smaller stress run while developing",
    )
    args = parser.parse_args()

    run_task_1(print_maps=args.maps, run_full_stress=not args.quick)
    run_task_2()
    run_task_3()


if __name__ == "__main__":
    main()
