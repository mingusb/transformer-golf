# Test Plan - challenger_1_stage13

This plan outline our steps to stress-test the generators in `src/data/context_sensitive.py` and verify correctness, limits, edge cases, type safety, and performance.

## Steps

1. **Investigate Existing Functionality & Boundaries**
   - Read `src/data/context_sensitive.py` and `tests/test_context_sensitive.py`.
   - Identify constraints (e.g., input type checks, sequence structure, complexity).

2. **Implement Stress Tests in `tests/test_context_sensitive_challenger_1.py`**
   - **Large Batch Scales**: Test `num_samples = 50,000` for both `generate_copy_task` and `generate_abc_task` (with fixed `n` and random `n`). Measure execution time and peak memory.
   - **Large Sequences**: Test `length = 2000` for copy task (total length 4001) and `n_max = 1000` for abc task (total length up to 3000). Check correctness and shapes.
   - **Minimal Parameters**: Test `length = 1, vocab_size = 2` for copy task, and `n_max = 1` for abc task (both fixed `n=1` and random `n=None`). Verify correctness of outputs.
   - **Type Checking and Coercion**: Verify that passing non-standard types like numpy integer types (e.g. `np.int64`, `np.int32`), float-like strings (e.g. `"5"`, `"5.0"`), and booleans (e.g. `True`, `False`) raise `ValueError`.
   - **Memory Leak/Excessive Allocation Analysis**: Run tests that generate multiple large batches sequentially and measure RSS memory growth to detect memory leaks.

3. **Execute Verification**
   - Run tests with `pytest tests/test_context_sensitive_challenger_1.py`.
   - Run existing tests `pytest tests/test_context_sensitive.py` to ensure zero regression.

4. **Analyze Results & Performance**
   - Analyze execution times, memory growth, and the behavior of the python loop inside `generate_abc_task` with `n=None`.
   - Check if there are any bugs, resource issues, or design flaws.

5. **Generate Reports**
   - Write a detailed `handoff.md` with:
     - Observations (verbatim metrics, commands used).
     - Logic Chain.
     - Caveats.
     - Conclusion (and Verdict: PASS/FAIL).
     - Verification Method.
   - Send completion message to parent Stage 13 Sub-Orchestrator.
