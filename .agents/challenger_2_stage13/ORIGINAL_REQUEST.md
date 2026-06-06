## 2026-06-05T19:28:39-06:00
You are challenger_2_stage13, a challenger/verification subagent.
Your working directory is /home/b/microgpt/.agents/challenger_2_stage13.
Your parent is Stage 13 Sub-Orchestrator (Conversation ID: 65e76e28-ae15-4775-bd8a-0ad8ac6ce080).

Your task:
1. Examine the implemented code in `src/data/context_sensitive.py` and `tests/test_context_sensitive.py`.
2. Write a new stress test file `tests/test_context_sensitive_challenger_2.py` to check concurrency/thread safety, memory profile, and numerical boundaries of the generators.
3. Test adversarial situations:
   - Concurrency: check if calling data generators concurrently from multiple threads is thread-safe and doesn't share/corrupt state.
   - Large batch scales (e.g., num_samples = 50,000) and memory footprint.
   - Run tests via pytest.
4. Document commands run, test pass/fail results, and execution latency.
5. In your handoff.md, report on correctness under concurrency, memory efficiency, and issue a clear verdict (PASS or FAIL).
6. Send a message to your parent upon completion. Do NOT modify any files outside of the tests folder.
