# Handoff Report: Stage 15 Explorer 1

## 1. Observation

1. **DualStackRNN Definition**: `src/models/universal_rnn.py` contains `class DualStackRNN(nn.Module)`. Its forward method signature is:
   ```python
   def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
   ```
   It returns a tuple of `(logits_tensor, (stack1_states_tensor, stack2_states_tensor))`.
2. **StackRNN Definition**: `src/models/stack_rnn.py` contains `class StackRNN(nn.Module)`. Its forward method signature is:
   ```python
   def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
   ```
   It returns a tuple of `(logits_tensor, stack_states_tensor)`.
3. **Current CLI options in `src/scripts/run_experiments.py`**:
   - Task options (lines 75):
     ```python
     parser.add_argument("--task", type=str, choices=["alternating", "nesting"], default="alternating")
     ```
   - Model options (line 76):
     ```python
     parser.add_argument("--model", type=str, default="all", choices=["all", "ssm", "attention", "conv1d", "markov", "stack_rnn", "lsm"])
     ```
4. **Context-Sensitive Data Generators in `src/data/context_sensitive.py`**:
   - `generate_copy_task(num_samples: int, length: int, vocab_size: int)` generates sequences of length `2 * length + 1` with a delimiter token at value `vocab_size - 1` and target padded with `0`.
   - `generate_abc_task(num_samples: int, n_max: int, n: int = None)` generates sequences of length `3 * n_max` using token `3` as padding/EOS. Target is left-shifted and padded with `3` at the end.
5. **Results Table CSV Path and Format**:
   - Defined in lines 340-347 of `run_experiments.py`:
     ```python
     table_path = os.path.join(args.output_dir, "results_table.csv")
     with open(table_path, "w") as f:
         f.write("model,accuracy,token_accuracy,sequence_accuracy,sparsity\n")
         for name in models_to_run:
             mean_token = np.mean(results[name]["token_accs"])
             mean_seq = np.mean(results[name]["seq_accs"])
             mean_sparsity = np.mean(results[name]["sparsities"])
             f.write(f"{name},{mean_token:.4f},{mean_token:.4f},{mean_seq:.4f},{mean_sparsity:.4f}\n")
     ```
6. **Device Mismatch in `sparsity.py`**:
   - In `apply_l0_mask` (line 97 of `src/models/sparsity.py`):
     ```python
     val_x = torch.randint(0, vocab_size, (10, 5))
     ```
     This tensor is constructed on CPU, which crashes if the model resides on a GPU/CUDA device during forward pass evaluation.

## 2. Logic Chain

1. **Importing & Model Integration (Addresses Point 1 & 2)**:
   - Dynamic imports using try-except blocks (consistent with `HAS_STACK_RNN`) will check for `DualStackRNN` and establish `HAS_DUAL_STACK_RNN = True/False`.
   - Adding `"dual_stack_rnn"` to argparse choices and mapping it in `model_map` inside `run_experiments.py` allows command-line target training.
2. **Dataset Generation & Configurations (Addresses Point 3 & 4)**:
   - For `copy`, vocabulary size must be at least `4` to accommodate three data tokens and one delimiter.
   - For `abc`, vocabulary size is exactly `4` because tokens are `0`, `1`, `2`, and the padding/EOS token is `3`.
   - To implement the new tasks, the general training/evaluation loop in `run_experiments.py` can be refactored to use a dynamically selected generator function based on the task choice.
3. **Evaluation Metrics (Addresses Point 5)**:
   - Python's sequence unpacking `logits, _ = model(X)` naturally accepts both `StackRNN` (returns `(logits, tensor)`) and `DualStackRNN` (returns `(logits, (tensor, tensor))`), requiring no changes to the logits extraction code.
   - The padding token (value `3`) in the `abc` task can skew token accuracy. Masking out padding tokens in the cross entropy loss (`ignore_index=3`) and using a custom boolean mask `Y != pad_token` in accuracy calculations prevents accuracy inflation.
4. **Saving Results (Addresses Point 6)**:
   - Writing to `results_table.csv` using the existing format `model,accuracy,token_accuracy,sequence_accuracy,sparsity` is required to maintain downstream compatibility.
5. **Device Placement (Addresses Point 7)**:
   - Tensors are natively on CPU. If a CUDA device is used, all datasets and models must be explicitly placed on the device via `.to(device)`.
   - A critical edge case exists in `sparsity.py` where verification dummy tensors are created on CPU. By moving the model to CPU before calling `apply_l0_mask`, transferring the output back to `device`, and setting `model.val_data = (X_val_on_device, Y_val_on_device)` before calling `l0_pruning_step`, we safely avoid CUDA runtime errors.

## 3. Caveats

- Assumes that any downstream scripts processing the generated `results_table.csv` do not expect specific tasks only, or that they are task-agnostic, which they appear to be.
- Assumes that the delimiter in the copy task (set to `vocab_size - 1 = 3`) is sufficient for the task without causing confusion with data tokens.

## 4. Conclusion

The implementation of Stage 15 (Evaluation & Integration) is feasible, cleanly integrates with the existing unified training loop structure, and can handle both the new models and tasks elegantly with appropriate masking and device-placement workarounds.

## 5. Verification Method

To verify the proposed implementation once implemented:
1. Run integration tests (such as `pytest tests/test_integration.py`).
2. Run the experiment suite in mock mode:
   ```bash
   python src/scripts/run_experiments.py --config mock --task copy --model all
   python src/scripts/run_experiments.py --config mock --task abc --model all
   ```
3. Inspect `results/results_table.csv` to ensure rows are populated for all models including `StackRNN` and `DualStackRNN`, and that columns are correctly formatted.
4. Invalidate if any CUDA device mismatch errors are thrown during execution.
