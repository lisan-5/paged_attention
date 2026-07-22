# PagedAttention Memory Simulator

This project is a pure Python simulation of KV-cache memory management. It does not use a GPU or a deep-learning library. Tokens, blocks, page tables, queues and GPU nodes are represented with normal Python data structures.

## What is included

### Task 1

- Naive contiguous allocator using first-fit placement
- Paged allocator with on-demand physical blocks
- Block sizes 4, 8, 16 and 32
- Same seed-42 workload for the comparison
- Per-step memory maps
- Internal and external fragmentation tracking
- 10,000 requests across five seeds
- Leak and ownership assertions
- Double-free, use-after-free and boundary-miss injection

### Task 2

- Oversubscribed preemption engine
- Recompute below threshold `T`
- Swap at or above threshold `T`
- Deterministic token generation and SHA-256 verification
- Priority and earliest-deadline scheduling
- Exact token-budget termination
- Continuous admission when slots open
- O(V+E) DFS deadlock detection
- Same-step deadlock recovery

### Task 3

- Copy-on-Write with reference counts
- Prefix caching by block-content hash
- Four-node distributed page table
- Spill to the node with the most free blocks
- Remote-block migration after more than three accesses in one step
- Node 2 failure at step 500 and request rescheduling

## Project layout

```text
models.py              shared request and result structures
workload.py            deterministic workload generation
naive_allocator.py     contiguous allocator and fragmentation
paged_allocator.py     physical blocks, page tables and stress suite
correctness.py         invariant checks and injected failures
scheduler.py           preemption and SLA scheduling
deadlock.py            wait-for graph and cycle recovery
bonus.py               Copy-on-Write, prefix cache and distribution
main.py                runs all required demonstrations
tests/test_project.py  focused unit tests
```

## Running it

No third-party packages are required.

```bash
python main.py
```

The default run includes the full 10,000-request, five-seed stress suite. To print the required per-step memory maps too:

```bash
python main.py --maps
```

For a faster development check:

```bash
python main.py --quick
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Workload choices

The project uses exponential inter-arrival times, which form a Poisson arrival process. Service times are also sampled exponentially. A request's actual token length is the sampled service time capped at its uniformly sampled maximum length.

The seed-42 comparison creates the workload once and deep-copies it for the two allocators. This prevents small random-number differences from making the comparison unfair.

## Fragmentation formulas

For the naive allocator:

```text
internal fragmentation = idle reserved slots / all reserved slots
```

A slot is idle when it belongs to an active request but has not yet been reached by generated tokens.

External fragmentation is measured when an allocation is attempted:

```text
external fragmentation = free slots in gaps smaller than the request / all free slots
```

The final value is the average of the per-step percentages. For the paged allocator, external fragmentation is zero because any free physical block can be used. Its internal fragmentation comes only from unused positions in currently allocated blocks, mostly the last block of each request.

## Important implementation details

A token at index `i` uses:

```python
logical_block = i // block_size
offset = i % block_size
```

A new block is allocated only when:

```python
logical_block >= len(page_table)
```

For `block_size = 4`, token 8 maps to logical block 2, offset 0. The tests explicitly check tokens 0, 3, 4 and 8.

The preemption demo generates each token from only the request ID and token index. Because the output is deterministic, a recomputed or swapped request can be compared with an uninterrupted baseline using SHA-256.

## Notes for the live explanation

The main design is intentionally simple:

- `set` stores free physical block IDs.
- `dict` stores page-table mappings and block contents.
- `deque` stores ordinary waiting requests.
- `heapq` implements `(priority, deadline, arrival order)` scheduling.
- DFS uses unvisited, visiting and visited states to find a back edge.

The code avoids frameworks and unnecessary abstractions so the allocation and scheduling decisions can be followed directly.
