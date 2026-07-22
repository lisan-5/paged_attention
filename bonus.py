"""Copy-on-Write KV-cache sharing implemented with block reference counts."""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple

from errors import OutOfMemoryError, UseAfterFreeError


class RefCountedAllocator:
    """Manage fixed-size blocks that may be shared by several completions."""

    def __init__(self, total_blocks: int, block_size: int):
        """Create an empty block pool and initialize every reference count."""
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks = set(range(total_blocks))
        self.data = {i: [None] * block_size for i in range(total_blocks)}
        self.ref_count = {i: 0 for i in range(total_blocks)}

    def allocate(self, request_id: str, refs: int = 1) -> int:
        """Allocate one block and set its initial number of owners."""
        if not self.free_blocks:
            raise OutOfMemoryError(request_id, "no free block")
        block = min(self.free_blocks)
        self.free_blocks.remove(block)
        self.data[block] = [None] * self.block_size
        self.ref_count[block] = refs
        return block

    def clone(self, block: int, request_id: str) -> int:
        """Create a private copy of an existing shared block."""
        self._check(block, request_id)
        new_block = self.allocate(request_id)
        self.data[new_block] = list(self.data[block])
        return new_block

    def decref(self, block: int) -> None:
        """Remove one owner and free the block when no owners remain."""
        self._check(block, "shared")
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self.data[block] = [None] * self.block_size
            self.free_blocks.add(block)

    def _check(self, block: int, request_id: str) -> None:
        """Reject attempts to access a block that has already been freed."""
        if block in self.free_blocks:
            raise UseAfterFreeError(request_id, f"block {block} is free")


@dataclass
class Completion:
    """Store one completion's page table, visible tokens, and lifecycle state."""

    completion_id: str
    page_table: Dict[int, int]
    tokens: List[str]
    cancelled: bool = False


class CopyOnWriteManager:
    """Share prompt blocks and create private copies only before modification."""

    def __init__(self, total_blocks: int = 128, block_size: int = 4):
        """Configure the reference-counted allocator and completion registry."""
        self.allocator = RefCountedAllocator(total_blocks, block_size)
        self.block_size = block_size
        self.completions: Dict[str, Completion] = {}

    def spawn(self, prompt: List[str], count: int) -> List[Completion]:
        """Create completions that initially share the prompt's physical blocks."""
        if count < 1:
            raise ValueError("count must be positive")

        shared = {}
        for logical, start in enumerate(range(0, len(prompt), self.block_size)):
            chunk = prompt[start : start + self.block_size]
            block = self.allocator.allocate("prompt", refs=count)
            self.allocator.data[block][: len(chunk)] = chunk
            shared[logical] = block

        result = []
        for index in range(count):
            completion = Completion(f"completion-{index}", dict(shared), list(prompt))
            self.completions[completion.completion_id] = completion
            result.append(completion)
        return result

    def write_token(self, completion: Completion, token: str) -> None:
        """Append a token, cloning its destination block if it is still shared."""
        if completion.cancelled:
            raise RuntimeError("completion is cancelled")

        logical, offset = divmod(len(completion.tokens), self.block_size)
        block = completion.page_table.get(logical)

        if block is None:
            block = self.allocator.allocate(completion.completion_id)
            completion.page_table[logical] = block
        elif self.allocator.ref_count[block] > 1:
            private = self.allocator.clone(block, completion.completion_id)
            self.allocator.decref(block)
            completion.page_table[logical] = block = private

        self.allocator.data[block][offset] = token
        completion.tokens.append(token)
        self.assert_isolation()

    def visible_tokens(self, completion: Completion) -> List[str]:
        """Read a completion's tokens through its logical-to-physical mappings."""
        visible = []
        for index in range(len(completion.tokens)):
            logical, offset = divmod(index, self.block_size)
            block = completion.page_table[logical]
            self.allocator._check(block, completion.completion_id)
            visible.append(self.allocator.data[block][offset])
        return visible

    def assert_isolation(self) -> None:
        """Verify that every active completion sees only its own token sequence."""
        for completion in self.completions.values():
            if not completion.cancelled:
                assert self.visible_tokens(completion) == completion.tokens

    def cancel(self, completion: Completion) -> None:
        """Release a completion's references and mark it as cancelled."""
        if completion.cancelled:
            return
        for block in completion.page_table.values():
            self.allocator.decref(block)
        completion.page_table.clear()
        completion.cancelled = True


class COWReport(NamedTuple):
    """Summarize tested branch counts, isolation failures, and block leaks."""

    tested_completion_counts: List[int]
    isolation_failures: int
    leaked_blocks: int


def run_cow_demo() -> COWReport:
    """Test Copy-on-Write isolation and cleanup with several branch counts."""
    tested = [1, 2, 4, 8]
    failures = 0

    for count in tested:
        manager = CopyOnWriteManager()
        completions = manager.spawn([f"prompt-{i}" for i in range(10)], count)

        for index, completion in enumerate(completions):
            manager.write_token(completion, f"branch-{index}")

        try:
            manager.assert_isolation()
        except AssertionError:
            failures += 1

        for completion in completions:
            manager.cancel(completion)
        assert len(manager.allocator.free_blocks) == manager.allocator.total_blocks

    return COWReport(tested, failures, 0)
