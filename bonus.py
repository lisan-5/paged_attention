"""Bonus simulations: Copy-on-Write, prefix reuse, and distributed paging."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from errors import OutOfMemoryError, UseAfterFreeError


class RefCountedBlockAllocator:
    """Manage shareable physical blocks whose lifetimes depend on reference counts."""

    def __init__(self, total_blocks: int, block_size: int):
        """Create empty blocks, a free pool, zeroed counts, and allocation metrics."""
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = set(range(total_blocks))
        self.data: Dict[int, List[Optional[str]]] = {
            block_id: [None] * block_size for block_id in range(total_blocks)
        }
        self.ref_count: Dict[int, int] = {block_id: 0 for block_id in range(total_blocks)}
        self.allocation_calls = 0

    def allocate(self, request_id: str, initial_ref_count: int = 1) -> int:
        """Allocate a clean block with the supplied initial number of references."""
        if not self.free_blocks:
            raise OutOfMemoryError(request_id, "no reference-counted block is available")
        block_id = min(self.free_blocks)
        self.free_blocks.remove(block_id)
        self.data[block_id] = [None] * self.block_size
        self.ref_count[block_id] = initial_ref_count
        self.allocation_calls += 1
        return block_id

    def incref(self, block_id: int) -> None:
        """Add one reader/owner to a live shared block."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError("shared", f"cannot incref free block {block_id}")
        self.ref_count[block_id] += 1

    def decref(self, block_id: int) -> None:
        """Drop one reference and free the block immediately when it reaches zero."""
        if block_id in self.free_blocks or self.ref_count[block_id] == 0:
            raise UseAfterFreeError("shared", f"cannot decref free block {block_id}")
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            self.data[block_id] = [None] * self.block_size
            self.free_blocks.add(block_id)

    def clone(self, block_id: int, request_id: str) -> int:
        """Create a private data copy for a completion that is about to write."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError(request_id, f"cannot clone free block {block_id}")
        new_block = self.allocate(request_id, initial_ref_count=1)
        self.data[new_block] = list(self.data[block_id])
        return new_block

    def read(self, block_id: int, offset: int, request_id: str) -> Optional[str]:
        """Read a live shared block at the requested token offset."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError(request_id, f"read from free block {block_id}")
        return self.data[block_id][offset]

    def write(self, block_id: int, offset: int, token: str, request_id: str) -> None:
        """Write to a live block after the manager has enforced Copy-on-Write."""
        if block_id in self.free_blocks:
            raise UseAfterFreeError(request_id, f"write to free block {block_id}")
        self.data[block_id][offset] = token


@dataclass
class Completion:
    """Represent one branch produced from a shared prompt."""

    completion_id: str
    page_table: Dict[int, int]
    tokens: List[str]
    cancelled: bool = False


class CopyOnWriteManager:
    """Share prompt blocks for reads and create private blocks before writes."""

    def __init__(self, total_blocks: int = 256, block_size: int = 4):
        """Create the reference-counted pool and completion registry."""
        self.allocator = RefCountedBlockAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.completions: Dict[str, Completion] = {}

    def spawn(self, prompt_tokens: List[str], count: int) -> List[Completion]:
        """Allocate one prompt copy and map it into N parallel completions."""
        if count <= 0:
            raise ValueError("count must be positive")

        base_page_table: Dict[int, int] = {}
        for logical_block, start in enumerate(range(0, len(prompt_tokens), self.block_size)):
            block_id = self.allocator.allocate("prompt", initial_ref_count=count)
            base_page_table[logical_block] = block_id
            for offset, token in enumerate(prompt_tokens[start : start + self.block_size]):
                self.allocator.write(block_id, offset, token, "prompt")

        spawned: List[Completion] = []
        for index in range(count):
            completion = Completion(
                completion_id=f"completion-{index}",
                page_table=dict(base_page_table),
                tokens=list(prompt_tokens),
            )
            self.completions[completion.completion_id] = completion
            spawned.append(completion)
        return spawned

    def write_token(self, completion: Completion, token: str) -> None:
        """Append a token, cloning a shared partial block before modification."""
        if completion.cancelled:
            raise RuntimeError("cannot write to a cancelled completion")

        token_index = len(completion.tokens)
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size

        if logical_block not in completion.page_table:
            block_id = self.allocator.allocate(completion.completion_id)
            completion.page_table[logical_block] = block_id
        else:
            block_id = completion.page_table[logical_block]
            if self.allocator.ref_count[block_id] > 1:
                cloned = self.allocator.clone(block_id, completion.completion_id)
                self.allocator.decref(block_id)
                completion.page_table[logical_block] = cloned
                block_id = cloned

        self.allocator.write(block_id, offset, token, completion.completion_id)
        completion.tokens.append(token)
        self.assert_isolation()

    def visible_tokens(self, completion: Completion) -> List[str]:
        """Reconstruct the tokens visible through one completion's page table."""
        visible: List[str] = []
        for token_index in range(len(completion.tokens)):
            logical_block = token_index // self.block_size
            offset = token_index % self.block_size
            block_id = completion.page_table[logical_block]
            value = self.allocator.read(block_id, offset, completion.completion_id)
            visible.append(value if value is not None else "")
        return visible

    def assert_isolation(self) -> None:
        """Prove every live completion sees exactly its own logical token sequence."""
        for completion in self.completions.values():
            if completion.cancelled:
                continue
            assert self.visible_tokens(completion) == completion.tokens

    def cancel(self, completion: Completion) -> None:
        """Release all references held by a cancelled completion."""
        if completion.cancelled:
            return
        for block_id in list(completion.page_table.values()):
            self.allocator.decref(block_id)
        completion.page_table.clear()
        completion.cancelled = True


@dataclass
class PrefixRequest:
    """Store a request whose leading blocks may come from the prefix cache."""

    request_id: str
    page_table: Dict[int, int]
    tokens: List[str]


class PrefixCache:
    """Reuse identical leading token blocks by content hash and reference count."""

    def __init__(self, total_blocks: int = 256, block_size: int = 4, minimum_prefix: int = 8):
        """Configure the shared allocator and minimum cacheable prefix length."""
        self.allocator = RefCountedBlockAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.minimum_prefix = minimum_prefix
        self.cache: Dict[str, int] = {}

    @staticmethod
    def _hash_chunk(tokens: List[str]) -> str:
        """Hash one token block using separators to avoid ambiguous concatenation."""
        payload = "\x1f".join(tokens).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _block_matches(self, block_id: int, chunk: List[str]) -> bool:
        """Verify content after a hash match to guard against collisions."""
        stored = self.allocator.data[block_id][: len(chunk)]
        return stored == chunk

    def add_request(self, request_id: str, tokens: List[str]) -> Tuple[PrefixRequest, int]:
        """Reuse the longest cached leading run, then allocate remaining blocks."""
        chunks = [tokens[index : index + self.block_size] for index in range(0, len(tokens), self.block_size)]

        cached_candidates: List[Tuple[int, int]] = []
        matched_tokens = 0
        for logical_block, chunk in enumerate(chunks):
            key = self._hash_chunk(chunk)
            block_id = self.cache.get(key)
            if block_id is None or block_id in self.allocator.free_blocks:
                break
            if not self._block_matches(block_id, chunk):
                break
            cached_candidates.append((logical_block, block_id))
            matched_tokens += len(chunk)

        use_cached_prefix = matched_tokens >= self.minimum_prefix
        page_table: Dict[int, int] = {}
        allocations_before = self.allocator.allocation_calls
        start_new_at = 0

        if use_cached_prefix:
            for logical_block, block_id in cached_candidates:
                self.allocator.incref(block_id)
                page_table[logical_block] = block_id
            start_new_at = len(cached_candidates)

        for logical_block in range(start_new_at, len(chunks)):
            chunk = chunks[logical_block]
            block_id = self.allocator.allocate(request_id)
            page_table[logical_block] = block_id
            for offset, token in enumerate(chunk):
                self.allocator.write(block_id, offset, token, request_id)
            self.cache[self._hash_chunk(chunk)] = block_id

        allocations = self.allocator.allocation_calls - allocations_before
        return PrefixRequest(request_id, page_table, list(tokens)), allocations

    def release(self, request: PrefixRequest) -> None:
        """Drop the request's references and free blocks whose counts reach zero."""
        for block_id in request.page_table.values():
            self.allocator.decref(block_id)
        request.page_table.clear()


@dataclass
class COWReport:
    """Summarize tested fan-out sizes, isolation errors, and leaked blocks."""

    tested_completion_counts: List[int]
    isolation_failures: int
    leaked_blocks: int


@dataclass
class PrefixReport:
    """Report whether the second request reused the full required prefix."""

    shared_prefix_tokens: int
    second_request_new_allocations: int
    success: bool


class GPUNode:
    """Simulate one independent GPU with its own physical block pool."""

    def __init__(self, node_id: int, total_blocks: int, block_size: int):
        """Create an online node whose blocks initially contain no tokens."""
        self.node_id = node_id
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.online = True
        self.free_blocks = set(range(total_blocks))
        self.data: Dict[int, List[Optional[str]]] = {
            block_id: [None] * block_size for block_id in range(total_blocks)
        }

    @property
    def free_count(self) -> int:
        """Report usable capacity, treating an offline node as having none."""
        return len(self.free_blocks) if self.online else 0

    def allocate(self, request_id: str) -> int:
        """Allocate a local block unless the node is offline or exhausted."""
        if not self.online or not self.free_blocks:
            raise OutOfMemoryError(request_id, f"node {self.node_id} has no free block")
        block_id = min(self.free_blocks)
        self.free_blocks.remove(block_id)
        self.data[block_id] = [None] * self.block_size
        return block_id

    def free(self, block_id: int) -> None:
        """Clear and return a block when its node is still online."""
        if not self.online:
            return
        self.data[block_id] = [None] * self.block_size
        self.free_blocks.add(block_id)

    def go_offline(self) -> None:
        """Simulate total node failure by making all local block data unavailable."""
        self.online = False
        self.free_blocks.clear()
        self.data.clear()


@dataclass
class DistributedRequest:
    """Represent a request whose page-table blocks may span several GPU nodes."""

    request_id: str
    actual_length: int
    local_node: int
    page_table: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    tokens_generated: int = 0
    token_sequence: List[str] = field(default_factory=list)
    status: str = "active"


@dataclass
class DistributedReport:
    """Summarize migration activity and recovery from the node-2 failure."""

    migrations: int
    node_failure_step: int
    affected_requests: int
    completed_requests: int
    hung_requests: int


class DistributedPageManager:
    """Route, access, migrate, and recover blocks across independent GPU nodes."""

    def __init__(self, node_block_counts: List[int], block_size: int = 8):
        """Create the GPU nodes and counters for per-step remote access frequency."""
        self.block_size = block_size
        self.nodes = {
            node_id: GPUNode(node_id, block_count, block_size)
            for node_id, block_count in enumerate(node_block_counts)
        }
        self.remote_accesses: Dict[Tuple[str, int, int], int] = {}
        self.migrations = 0

    def _choose_node(self, local_node: int, request_id: str) -> GPUNode:
        """Prefer local allocation, otherwise spill to the node with most space."""
        local = self.nodes[local_node]
        if local.online and local.free_count:
            return local

        candidates = [node for node in self.nodes.values() if node.online and node.free_count]
        if not candidates:
            raise OutOfMemoryError(request_id, "all online nodes are full")
        return max(candidates, key=lambda node: (node.free_count, -node.node_id))

    def allocate_for(self, request: DistributedRequest, logical_block: int) -> Tuple[int, int]:
        """Allocate one logical block and store its `(node, block)` location."""
        node = self._choose_node(request.local_node, request.request_id)
        block_id = node.allocate(request.request_id)
        request.page_table[logical_block] = (node.node_id, block_id)
        return node.node_id, block_id

    def write_token(self, request: DistributedRequest, token: str) -> None:
        """Allocate on demand and write the next token on the mapped node."""
        token_index = request.tokens_generated
        logical_block = token_index // self.block_size
        offset = token_index % self.block_size
        if logical_block not in request.page_table:
            self.allocate_for(request, logical_block)

        node_id, block_id = request.page_table[logical_block]
        node = self.nodes[node_id]
        if not node.online:
            raise UseAfterFreeError(request.request_id, f"node {node_id} is offline")
        node.data[block_id][offset] = token
        request.token_sequence.append(token)
        request.tokens_generated += 1

    def access_block(self, request: DistributedRequest, logical_block: int, step: int) -> List[Optional[str]]:
        """Read a block and migrate it after more than three same-step remote reads."""
        node_id, block_id = request.page_table[logical_block]
        node = self.nodes[node_id]
        if not node.online:
            raise UseAfterFreeError(request.request_id, f"node {node_id} is offline")

        if node_id != request.local_node:
            key = (request.request_id, logical_block, step)
            self.remote_accesses[key] = self.remote_accesses.get(key, 0) + 1
            if self.remote_accesses[key] > 3:
                self._migrate(request, logical_block)
                node_id, block_id = request.page_table[logical_block]
                node = self.nodes[node_id]

        return list(node.data[block_id])

    def _make_local_space(self, request: DistributedRequest, target_logical: int) -> bool:
        """Spill a colder local block when a hot remote block needs local space."""
        local = self.nodes[request.local_node]
        if local.free_count:
            return True

        # Spill one colder local block belonging to this request. This keeps the
        # hot-block migration transparent even when the local pool is full.
        spill_candidates = [
            (logical_block, block_id)
            for logical_block, (node_id, block_id) in request.page_table.items()
            if node_id == request.local_node and logical_block != target_logical
        ]
        destinations = [
            node
            for node in self.nodes.values()
            if node.online and node.node_id != request.local_node and node.free_count
        ]
        if not spill_candidates or not destinations:
            return False

        logical_to_spill, local_block = spill_candidates[0]
        destination = max(destinations, key=lambda node: (node.free_count, -node.node_id))
        remote_block = destination.allocate(request.request_id)
        destination.data[remote_block] = list(local.data[local_block])
        request.page_table[logical_to_spill] = (destination.node_id, remote_block)
        local.free(local_block)
        return True

    def _migrate(self, request: DistributedRequest, logical_block: int) -> None:
        """Copy a remote block locally, update its mapping, and free the source."""
        source_node_id, source_block = request.page_table[logical_block]
        if source_node_id == request.local_node:
            return
        local = self.nodes[request.local_node]
        if not local.online or not self._make_local_space(request, logical_block):
            return

        source = self.nodes[source_node_id]
        new_block = local.allocate(request.request_id)
        local.data[new_block] = list(source.data[source_block])
        request.page_table[logical_block] = (local.node_id, new_block)
        source.free(source_block)
        self.migrations += 1

    def release(self, request: DistributedRequest) -> None:
        """Return all surviving distributed blocks owned by one request."""
        for node_id, block_id in list(request.page_table.values()):
            node = self.nodes[node_id]
            if node.online:
                node.free(block_id)
        request.page_table.clear()

    def fail_node(self, node_id: int, requests: List[DistributedRequest]) -> List[str]:
        """Fail a node and reset every affected request for recomputation elsewhere."""
        affected = [
            request
            for request in requests
            if any(location[0] == node_id for location in request.page_table.values())
        ]

        self.nodes[node_id].go_offline()
        for request in affected:
            for stored_node_id, block_id in list(request.page_table.values()):
                if stored_node_id != node_id and self.nodes[stored_node_id].online:
                    self.nodes[stored_node_id].free(block_id)
            request.page_table.clear()
            request.tokens_generated = 0
            request.token_sequence.clear()
            request.status = "waiting"
        return [request.request_id for request in affected]


def run_cow_demo() -> COWReport:
    """Test sharing and branch isolation for N equal to 1, 2, 4, and 8."""
    failures = 0
    tested = [1, 2, 4, 8]
    for count in tested:
        manager = CopyOnWriteManager(total_blocks=128, block_size=4)
        prompt = [f"prompt-{index}" for index in range(10)]
        completions = manager.spawn(prompt, count)
        for index, completion in enumerate(completions):
            manager.write_token(completion, f"branch-{index}-a")
            manager.write_token(completion, f"branch-{index}-b")
        try:
            manager.assert_isolation()
        except AssertionError:
            failures += 1
        for completion in completions:
            manager.cancel(completion)
        assert len(manager.allocator.free_blocks) == manager.allocator.total_blocks

    return COWReport(tested, failures, 0)


def run_prefix_demo() -> PrefixReport:
    """Prove a second 20-token prefix needs exactly zero allocations."""
    cache = PrefixCache(total_blocks=64, block_size=4, minimum_prefix=8)
    shared_prefix = [f"shared-{index}" for index in range(20)]
    first, _ = cache.add_request("first", shared_prefix + ["first-tail"])
    second, second_allocations = cache.add_request("second", shared_prefix)

    success = second_allocations == 0
    assert success

    cache.release(second)
    cache.release(first)
    return PrefixReport(20, second_allocations, success)


def run_distributed_demo() -> DistributedReport:
    """Demonstrate hot-block migration and recovery from node 2 at step 500."""
    # First demonstrate transparent migration after four remote accesses.
    migration_manager = DistributedPageManager([3, 4, 4, 4], block_size=4)
    dummy_block = migration_manager.nodes[0].allocate("dummy")
    migration_request = DistributedRequest("migration", actual_length=12, local_node=0)
    for index in range(12):
        migration_manager.write_token(migration_request, f"m-{index}")
    remote_logical = next(
        logical
        for logical, (node_id, _) in migration_request.page_table.items()
        if node_id != migration_request.local_node
    )
    migration_manager.nodes[0].free(dummy_block)
    for _ in range(4):
        migration_manager.access_block(migration_request, remote_logical, step=10)
    assert migration_request.page_table[remote_logical][0] == 0
    migration_manager.release(migration_request)

    # Then run through step 500 and fail node 2.
    manager = DistributedPageManager([128, 128, 128, 128], block_size=8)
    requests = [
        DistributedRequest(
            request_id=f"distributed-{index}",
            actual_length=510,
            local_node=index % 4,
        )
        for index in range(8)
    ]

    failure_step = 500
    affected_ids: List[str] = []
    completed = 0
    max_steps = 1_500

    for step in range(max_steps):
        if step == failure_step:
            affected_ids = manager.fail_node(2, requests)

        for request in requests:
            if request.status == "completed":
                continue
            try:
                token = f"{request.request_id}-token-{request.tokens_generated}"
                manager.write_token(request, token)
                request.status = "active"
            except OutOfMemoryError:
                request.status = "waiting"
                continue

            if request.tokens_generated >= request.actual_length:
                manager.release(request)
                request.status = "completed"
                completed += 1

        if completed == len(requests):
            break

    hung = sum(request.status != "completed" for request in requests)
    assert affected_ids
    assert hung == 0

    return DistributedReport(
        migrations=migration_manager.migrations,
        node_failure_step=failure_step,
        affected_requests=len(affected_ids),
        completed_requests=completed,
        hung_requests=hung,
    )
