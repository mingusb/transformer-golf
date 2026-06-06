## 2026-06-05T19:32:10Z
You are teamwork_preview_worker. Your task is to implement the `DualStackRNN` class in `/home/b/microgpt/src/models/universal_rnn.py` according to the design proposal located at `/home/b/microgpt/.agents/sub_orch_stage14/design_proposal.md`.
Please write the code and run the build and tests to verify it works. Specifically, run the following command to check your implementation against the unit tests:
`pytest tests/test_phase_3.py -k "DualStackRNN"`
Ensure that all implementation details match the design proposal exactly. Expose stack_width and stack_depth on the class/instance, support .to(device) correctly, raise ValueError on dimensions <= 0, and return the correct shapes: logits shape (batch_size, seq_len, vocab_size) and stack states tuple ((batch_size, seq_len, stack_depth, stack_width), (batch_size, seq_len, stack_depth, stack_width)).
Write your handoff report to `/home/b/microgpt/.agents/worker_implement/handoff.md`.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.
