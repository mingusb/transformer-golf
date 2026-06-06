## 2026-06-05T19:24:27-06:00
You are worker_stage13, a developer worker subagent.
Your working directory is /home/b/microgpt/.agents/worker_stage13.
Your parent is Stage 13 Sub-Orchestrator (Conversation ID: 65e76e28-ae15-4775-bd8a-0ad8ac6ce080).

Your task:
1. Read the design specification in /home/b/microgpt/.agents/explorer_stage13/design_proposal.md.
2. Implement the generators:
   - generate_copy_task(num_samples: int, length: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]
   - generate_abc_task(num_samples: int, n_max: int, n: int = None) -> Tuple[torch.Tensor, torch.Tensor]
   in `src/data/context_sensitive.py`.
3. Ensure that:
   - Inputs and targets are torch.long tensors.
   - Targets are left-shifted inputs (e.g., targets = inputs[:, 1:] with appropriate padding at the end).
   - In generate_copy_task, copy sequence w delimiter w has total length 2 * length + 1. The delimiter token is vocab_size - 1. Content tokens are in range [0, vocab_size - 2]. The target sequence is left-shifted input with last token 0.
   - In generate_abc_task, if n is None, n is sampled uniformly from [1, n_max] per sample, and sequences are right-padded to 3 * n_max with PAD token 3. If n is specified, sequences have length exactly 3 * n with no padding. The target sequence is left-shifted input with last token 3.
   - Inputs are validated and raise ValueError for invalid parameters.
4. Implement a comprehensive unit test suite in `tests/test_context_sensitive.py` that checks the properties (shapes, dtypes, values, delimiter placement, padding, invalid parameter handling) of the generators.
5. Verify the code and tests by running the test suite using pytest. Ensure all tests pass.
6. Write a handoff.md in your working directory summarizing:
   - What changes you made (list files created/modified)
   - The exact pytest command and its terminal output verifying all tests passed
   - Any comments on implementation details or assumptions.
7. Send a message to your parent upon completion.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
