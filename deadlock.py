"""Wait-for graph, linear-time cycle detection, and deadlock recovery demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class DeadlockRequest:
    """Store the state needed to choose a victim from a deadlock cycle."""

    request_id: str
    held_blocks: Set[int]
    tokens_generated: int
    status: str = "active"


@dataclass
class DeadlockReport:
    """Describe the detected cycle, chosen victim, and recovery result."""

    cycle_detected: bool
    cycle: List[str]
    victim: Optional[str]
    recovered_same_step: bool
    remaining_edges: int


class WaitForGraph:
    """Model `A waits for B` as a directed edge from request A to B."""

    def __init__(self):
        """Create an empty adjacency-list representation of the graph."""
        self.edges: Dict[str, Set[str]] = {}

    def add_request(self, request_id: str) -> None:
        """Add a request node if it is not already present."""
        self.edges.setdefault(request_id, set())

    def add_wait(self, waiting_request: str, holder_request: str) -> None:
        """Record that one request is blocked by a resource holder."""
        self.add_request(waiting_request)
        self.add_request(holder_request)
        self.edges[waiting_request].add(holder_request)

    def remove_request(self, request_id: str) -> None:
        """Remove a victim and every edge entering or leaving that victim."""
        self.edges.pop(request_id, None)
        for neighbours in self.edges.values():
            neighbours.discard(request_id)

    def find_cycle(self) -> List[str]:
        """Return one directed cycle using O(V+E) DFS."""
        state: Dict[str, int] = {node: 0 for node in self.edges}
        parent: Dict[str, Optional[str]] = {node: None for node in self.edges}

        def dfs(node: str) -> List[str]:
            """Search one DFS branch and return the first back-edge cycle found."""
            state[node] = 1
            for neighbour in self.edges.get(node, set()):
                if state[neighbour] == 0:
                    parent[neighbour] = node
                    found = dfs(neighbour)
                    if found:
                        return found
                elif state[neighbour] == 1:
                    cycle = [neighbour]
                    cursor = node
                    while cursor != neighbour:
                        cycle.append(cursor)
                        cursor = parent[cursor]  # type: ignore[index]
                    cycle.reverse()
                    return cycle
            state[node] = 2
            return []

        for node in self.edges:
            if state[node] == 0:
                cycle = dfs(node)
                if cycle:
                    return cycle
        return []


class DeadlockManager:
    """Combine request metadata with the wait-for graph recovery policy."""

    def __init__(self):
        """Initialize the request registry and its empty wait-for graph."""
        self.requests: Dict[str, DeadlockRequest] = {}
        self.graph = WaitForGraph()

    def add_request(self, request: DeadlockRequest) -> None:
        """Register a request in both the metadata store and graph."""
        self.requests[request.request_id] = request
        self.graph.add_request(request.request_id)

    def recover(self) -> DeadlockReport:
        """Detect a cycle and evict the prescribed victim in the same call."""
        cycle = self.graph.find_cycle()
        if not cycle:
            return DeadlockReport(False, [], None, True, self.edge_count())

        cycle_requests = [self.requests[request_id] for request_id in cycle]
        victim = min(
            cycle_requests,
            key=lambda request: (-len(request.held_blocks), request.tokens_generated, request.request_id),
        )

        victim.held_blocks.clear()
        victim.status = "preempted"
        self.graph.remove_request(victim.request_id)

        recovered = not self.graph.find_cycle()
        return DeadlockReport(
            cycle_detected=True,
            cycle=cycle,
            victim=victim.request_id,
            recovered_same_step=recovered,
            remaining_edges=self.edge_count(),
        )

    def edge_count(self) -> int:
        """Return the number of active waiting relationships in the graph."""
        return sum(len(neighbours) for neighbours in self.graph.edges.values())


def run_deadlock_demo(print_output: bool = True) -> DeadlockReport:
    """Construct a guaranteed A-B cycle and prove same-step recovery."""
    manager = DeadlockManager()
    request_a = DeadlockRequest("A", held_blocks={7, 8}, tokens_generated=9)
    request_b = DeadlockRequest("B", held_blocks={12}, tokens_generated=4)
    manager.add_request(request_a)
    manager.add_request(request_b)

    manager.graph.add_wait("A", "B")
    manager.graph.add_wait("B", "A")

    report = manager.recover()
    assert report.cycle_detected
    assert report.victim == "A"  # A holds the most blocks.
    assert report.recovered_same_step

    if print_output:
        print(f"Deadlock cycle detected: {' -> '.join(report.cycle)}")
        print(f"Evicted request: {report.victim}")
        print(f"Recovered in same step: {report.recovered_same_step}")

    return report
