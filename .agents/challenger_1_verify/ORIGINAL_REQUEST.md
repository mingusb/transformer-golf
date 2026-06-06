## 2026-06-06T01:35:04Z
You are teamwork_preview_challenger. Your task is to stress-test the `DualStackRNN` model in `src/models/universal_rnn.py`.
1. Write a script/test case to evaluate the model with extreme input sizes (large sequence lengths, large batch sizes, large dimensions) to verify it does not produce NaN or overflow under standard gradient steps.
2. Run tests to confirm gradients are flowing to all layers.
3. Run the unit tests:
   `pytest tests/test_phase_3.py`
4. Report your stress test methodology and results in `/home/b/microgpt/.agents/challenger_1_verify/handoff.md`.
