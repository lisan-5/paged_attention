"""Deterministic workload generators shared by all simulation tasks."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import DefaultDict, Iterable, List

from models import Request


def generate_workload(
    count: int,
    seed: int,
    arrival_rate: float = 0.9,
    mean_service_time: float = 70.0,
    min_length: int = 10,
    max_length: int = 512,
) -> List[Request]:
    """Create a deterministic Poisson-arrival/exponential-service workload.

    Exponential inter-arrival times produce a Poisson arrival process. Service
    time is also exponentially distributed and is used as the actual token
    length, capped by the request's uniformly sampled maximum length.
    """
    rng = random.Random(seed)
    requests: List[Request] = []
    arrival_clock = 0.0

    for request_id in range(count):
        if request_id > 0:
            arrival_clock += rng.expovariate(arrival_rate)

        sampled_service = max(1, int(round(rng.expovariate(1.0 / mean_service_time))))
        sampled_max = rng.randint(min_length, max_length)
        actual_length = min(sampled_service, sampled_max)

        requests.append(
            Request(
                request_id=request_id,
                arrival_time=int(arrival_clock),
                max_length=sampled_max,
                actual_length=actual_length,
                service_time=sampled_service,
            )
        )

    return requests


def generate_scheduler_workload(count: int = 500, seed: int = 2025) -> List[Request]:
    """Create requests with priorities, deadlines, and exact token budgets."""
    rng = random.Random(seed)
    requests: List[Request] = []
    arrival_clock = 0.0

    for request_id in range(count):
        if request_id > 0:
            arrival_clock += rng.expovariate(3.0)

        actual_length = rng.randint(10, 40)
        max_length = rng.randint(actual_length, 64)
        budget = rng.randint(6, 26)

        requests.append(
            Request(
                request_id=request_id,
                arrival_time=int(arrival_clock),
                max_length=max_length,
                actual_length=actual_length,
                service_time=actual_length,
                priority=rng.randint(1, 3),
                deadline=rng.randint(12, 45),
                token_budget=budget,
            )
        )

    return requests


def clone_workload(requests: Iterable[Request]) -> List[Request]:
    """Deep-copy requests and reset state for an apples-to-apples comparison."""
    copied = copy.deepcopy(list(requests))
    for request in copied:
        request.reset_runtime()
    return copied


def group_by_arrival(requests: Iterable[Request]) -> DefaultDict[int, List[Request]]:
    """Index requests by arrival step so each simulation can admit them quickly."""
    grouped: DefaultDict[int, List[Request]] = defaultdict(list)
    for request in requests:
        grouped[request.arrival_time].append(request)
    return grouped
