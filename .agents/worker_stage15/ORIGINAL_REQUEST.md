## 2026-06-05T19:59:56Z
Implement Stage 15 (Evaluation & Integration) in src/scripts/run_experiments.py.
Your identity: Stage 15 Worker 1.
Your working directory: /home/b/microgpt/.agents/worker_stage15

Specifically, execute the following:
1. Update src/scripts/run_experiments.py to import DualStackRNN from src.models.universal_rnn inside a try-except block, defining HAS_DUAL_STACK_RNN.
2. Update the argparse parser to support choices=["alternating", "nesting", "copy", "abc"] for --task and add "dual_stack_rnn" choice to --model.
3. In the task selection block, add copy and abc tasks. Set up task-specific configurations:
   - For copy task:
     - Mock config: seeds=[1], epochs=2, train_len=5, test_len=10, vocab_size=4, num_samples=5.
     - Real config: seeds=list(range(1, 11)), epochs=80, train_len=20, test_len=100, vocab_size=10, num_samples=100.
   - For abc task:
     - Mock config: seeds=[1], epochs=2, train_len=5, test_len=10, vocab_size=4, num_samples=5.
     - Real config: seeds=list(range(1, 11)), epochs=80, train_len=20, test_len=100, vocab_size=4, num_samples=100.
4. Support training and evaluating DualStackRNN vs StackRNN on copy and abc tasks under length generalization.
   - For copy task: use generate_copy_task from src.data.context_sensitive. ignore_index = None.
   - For abc task: use generate_abc_task from src.data.context_sensitive. Use fixed count n (i.e. n=train_len and n=test_len respectively) to enforce length generalization. ignore_index = 3.
5. Enhance train_model and evaluate_model_accs to accept an optional ignore_index parameter. Ensure evaluation sequence accuracy correctly masks out ignore_index tokens (e.g. check only non-ignored positions in the sequence).
6. Save the comparison metrics (token accuracy, sequence accuracy, sparsity) to results_table.csv under the specified output directory in the format:
   model,accuracy,token_accuracy,sequence_accuracy,sparsity
   (Note: accuracy and token_accuracy should both contain the mean token accuracy. Sparsity for both StackRNN and DualStackRNN is 0.0000. All metrics formatted to 4 decimal places).
7. Ensure StackRNN sequence accuracy is < 1.0 on copy/abc tasks while DualStackRNN achieves high accuracy (or 1.0 sequence accuracy on abc).
8. Run pytest tests (e.g. pytest tests/test_phase_3.py) and/or direct script executions (using --config mock) to verify your modifications are correct and all tests pass.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Write your changes report to /home/b/microgpt/.agents/worker_stage15/changes.md and handoff report to /home/b/microgpt/.agents/worker_stage15/handoff.md. Include the test outputs and commands run in your handoff message.
