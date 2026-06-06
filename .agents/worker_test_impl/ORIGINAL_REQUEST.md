## 2026-06-06T01:25:20Z

Your task is to implement the E2E test suite in `/home/b/microgpt/tests/test_phase_3.py`.
The test suite must cover Tiers 1-4 for Phase 3 (Stages 13, 14, 15) with features:
- F1: Copy Task Generator
- F2: ABC Task Generator
- F3: DualStackRNN
- F4: run_experiments.py

Specifically, implement the following tests in `/home/b/microgpt/tests/test_phase_3.py`:

1. Dynamic Imports:
At the top of `tests/test_phase_3.py`, dynamically attempt to import `generate_copy_task` and `generate_abc_task` from `src.data.context_sensitive`, and `DualStackRNN` from `src.models.universal_rnn`.
Set flags `HAS_COPY_DATA`, `HAS_ABC_DATA`, and `HAS_DUAL_STACK_RNN` to True/False accordingly.
If an import fails, catch the `ImportError` and set the flag to False.
Use `@pytest.mark.skipif(...)` on the tests that depend on these imports so that they can be run and skipped gracefully if the implementation has not yet been built.
Also import standard Python libraries needed: `pytest`, `torch`, `torch.nn as nn`, `numpy as np`, `os`, `sys`, `subprocess`.

2. Tier 1 Tests (Feature Coverage):
Implement 5 tests for each feature (Total 20 tests):
- Feature F1 (Copy Task Generator):
  - `test_TEST_T1_F1_01_generation_shapes`
  - `test_TEST_T1_F1_02_vocab_range`
  - `test_TEST_T1_F1_03_delimiter_position`
  - `test_TEST_T1_F1_04_target_mapping`
  - `test_TEST_T1_F1_05_copy_property`
- Feature F2 (ABC Task Generator):
  - `test_TEST_T1_F2_01_generation_shapes`
  - `test_TEST_T1_F2_02_vocab_range`
  - `test_TEST_T1_F2_03_structure`
  - `test_TEST_T1_F2_04_target_mapping`
  - `test_TEST_T1_F2_05_various_n`
- Feature F3 (DualStackRNN):
  - `test_TEST_T1_F3_01_initialization`
  - `test_TEST_T1_F3_02_forward_shape`
  - `test_TEST_T1_F3_03_stack_operations`
  - `test_TEST_T1_F3_04_differentiability`
  - `test_TEST_T1_F3_05_stack_dimensions`
- Feature F4 (run_experiments.py):
  - `test_TEST_T1_F4_01_cli_copy_task`
  - `test_TEST_T1_F4_02_cli_abc_task`
  - `test_TEST_T1_F4_03_output_csv_existence`
  - `test_TEST_T1_F4_04_output_plot_existence`
  - `test_TEST_T1_F4_05_cli_model_choice`

3. Tier 2 Tests (Boundary & Corner cases):
Implement 5 tests for each feature (Total 20 tests):
- Feature F1 (Copy Task Generator):
  - `test_TEST_T2_F1_01_invalid_num_samples`
  - `test_TEST_T2_F1_02_invalid_seq_len_even`
  - `test_TEST_T2_F1_03_invalid_seq_len_negative`
  - `test_TEST_T2_F1_04_invalid_vocab_size`
  - `test_TEST_T2_F1_05_reproducibility`
- Feature F2 (ABC Task Generator):
  - `test_TEST_T2_F2_01_invalid_num_samples`
  - `test_TEST_T2_F2_02_invalid_seq_len_not_div_3`
  - `test_TEST_T2_F2_03_invalid_seq_len_negative`
  - `test_TEST_T2_F2_04_invalid_vocab_size`
  - `test_TEST_T2_F2_05_reproducibility`
- Feature F3 (DualStackRNN):
  - `test_TEST_T2_F3_01_zero_stack_width`
  - `test_TEST_T2_F3_02_min_vocab_size`
  - `test_TEST_T2_F3_03_extreme_inputs`
  - `test_TEST_T2_F3_04_device_transfer`
  - `test_TEST_T2_F3_05_gradient_clip`
- Feature F4 (run_experiments.py):
  - `test_TEST_T2_F4_01_invalid_task`
  - `test_TEST_T2_F4_02_invalid_model`
  - `test_TEST_T2_F4_03_empty_output_dir`
  - `test_TEST_T2_F4_04_mock_config_epochs`
  - `test_TEST_T2_F4_05_overwriting_outputs`

4. Tier 3 Tests (Pairwise integration):
Implement at least 4 tests:
- `test_TEST_T3_01_generator_and_model_copy`
- `test_TEST_T3_02_generator_and_model_abc`
- `test_TEST_T3_03_cli_runs_copy_with_dual_stack`
- `test_TEST_T3_04_cli_runs_abc_with_dual_stack`

5. Tier 4 Tests (Real-world workload / application scenarios):
Implement at least 5 tests:
- `test_TEST_T4_01_copy_memorization`
- `test_TEST_T4_02_abc_memorization`
- `test_TEST_T4_03_copy_generalization`
- `test_TEST_T4_04_abc_generalization`
- `test_TEST_T4_05_baseline_comparison`

Please implement these tests in `/home/b/microgpt/tests/test_phase_3.py`.
Run `.venv/bin/pytest tests/test_phase_3.py` to ensure that all written tests compile and run, and are skipped gracefully.
Update your `progress.md` before and after doing the work.
Once you're done, write `handoff.md` in your working directory and notify the parent orchestrator via `send_message` with your results.
