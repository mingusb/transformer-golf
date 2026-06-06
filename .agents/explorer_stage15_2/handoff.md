# Handoff Report - Stage 15 Explorer 2

## 1. Observation
We observed the following details in the codebase:
- **File Paths and Lines**:
  - `src/scripts/run_experiments.py` (lines 75-76): CLI parser currently restricts tasks to `choices=["alternating", "nesting"]` and models to `choices=["all", "ssm", "attention", "conv1d", "markov", "stack_rnn", "lsm"]`.
  - `src/models/universal_rnn.py` (line 32, 127): `DualStackRNN` forward pass has signature `def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]` and returns `logits_tensor, (stack1_states_tensor, stack2_states_tensor)`.
  - `src/data/context_sensitive.py` (lines 4-8): `generate_copy_task` signature expects `num_samples`, `length`, and `vocab_size`.
  - `src/data/context_sensitive.py` (lines 58-62, 107-112): `generate_abc_task` signature expects `num_samples`, `n_max`, and optional `n`. The padding/EOS token is fixed as `3` and sequence values are in `[0, 1, 2, 3]`, indicating the task vocabulary size is exactly 4.
  - `src/models/sparsity.py` (lines 97-100): Inside `apply_l0_mask`, validation tensors are created without device placement:
    ```python
    val_x = torch.randint(0, vocab_size, (10, 5))
    val_y = torch.randint(0, vocab_size, (10, 5))
    with torch.no_grad():
        out = model(val_x)
    ```
    If `model` is already transferred to GPU (CUDA), this causes a `RuntimeError` due to device mismatch.

## 2. Logic Chain
1. **Unpacking Compatibility**: Because `evaluate_model_accs` and `train_model` unpack model returns as `logits, _ = model(X)`, and Python's `_` discard variable can bind to any object (including a tuple of stack state tensors), the output structure of `DualStackRNN` is compatible out-of-the-box.
2. **Vocabulary Size and Padding**: Since `generate_abc_task` outputs sequences using token `3` as padding/EOS, the vocabulary size is exactly 4. Setting the model's `vocab_size` to 4 is necessary. Additionally, to avoid accuracy distortion from padding when `n = None`, target padding tokens (`3`) should be ignored using a mask when computing token and sequence accuracies.
3. **Device Mismatch Risk**: The CPU-bound initialization of `val_x` and `val_y` in `apply_l0_mask` will crash if the model is moved to GPU beforehand. Therefore, running experiments on CPU by default is the safest and most performant solution, given that the models are tiny.
4. **Integration**: Updating CLI arguments, tasks, and baseline routing enables calling `run_experiments.py` with `--task copy` and `--task abc` as required by the phase tests.

## 3. Caveats
- We assume that the user's test suite does not enforce running on CUDA, as CPU performance is more than sufficient for these tiny RNNs.
- The default behavior of `generate_abc_task` with `n` specified (which produces sequences of exact length `3 * n` with no internal padding) is preferred for length generalization evaluation, as it provides a clean generalization signal.

## 4. Conclusion
The implementation of Stage 15 can be cleanly achieved by:
1. Dynamically importing `DualStackRNN` and setting the flag `HAS_DUAL_STACK_RNN`.
2. Expanding the CLI arguments to choices `["alternating", "nesting", "copy", "abc"]` and model choices to include `dual_stack_rnn`.
3. Creating training and evaluation configurations matching the mock/real scaling parameters.
4. Using masked accuracy computations in evaluation loops to handle potential padding tokens.
5. Saving results to `output_dir/results_table.csv` using the columns `model,accuracy,token_accuracy,sequence_accuracy,sparsity`.
6. Running computations on CPU to avoid device mismatch crashes in the L0 pruning steps.

## 5. Verification Method
Verify that the design works by running:
```bash
pytest tests/test_phase_3.py
```
This test suite requires `run_experiments.py` to support CLI task options and execute `copy` and `abc` tasks with the `dual_stack_rnn` model.
To run the verification on the updated script, execute:
```bash
python src/scripts/run_experiments.py --task copy --config mock --model dual_stack_rnn
python src/scripts/run_experiments.py --task abc --config mock --model dual_stack_rnn
```
Verify the output table is correctly formatted at `results/results_table.csv`.
