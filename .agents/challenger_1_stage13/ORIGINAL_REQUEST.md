## 2026-06-05T19:28:39-06:00
You are challenger_1_stage13, a challenger/verification subagent.
Your working directory is /home/b/microgpt/.agents/challenger_1_stage13.
Your parent is Stage 13 Sub-Orchestrator (Conversation ID: 65e76e28-ae15-4775-bd8a-0ad8ac6ce080).

Your task:
1. Examine the implemented code in `src/data/context_sensitive.py` and `tests/test_context_sensitive.py`.
2. Write a new stress test file `tests/test_context_sensitive_challenger_1.py` to empirically stress-test the generators.
3. Test edge cases, extreme inputs, type coercion, and performance:
   - Large batch scales (e.g., num_samples = 50,000)
   - Large sequences (length = 2000, n_max = 1000)
   - Minimal parameters (length = 1, vocab_size = 2, n_max = 1)
   - Invalid arguments passed as non-standard types (e.g. numpy integer types, float-like strings, booleans).
   - Check if there are memory leaks or excessive allocation.
4. Execute the tests via pytest and document findings.
5. In your handoff.md, report on execution times, resource consumption, edge-case coverage, and write a clear verdict (PASS or FAIL).
6. Send a message to your parent upon completion. Do NOT modify any files outside of the tests folder.
