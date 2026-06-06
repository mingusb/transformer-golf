# Changes made by Stage 15 Worker 1

## Summary of Modifications
All modifications were restricted to `src/scripts/run_experiments.py`.

### 1. DualStackRNN Import and Flag Definition
- Imported `DualStackRNN` from `src.models.universal_rnn` inside a `try-except` block to define `HAS_DUAL_STACK_RNN`.

### 2. Argument Parser Enhancements
- Updated argparse parser choices for `--task` to include `["alternating", "nesting", "copy", "abc"]`.
- Added `"dual_stack_rnn"` to choices for `--model`.

### 3. Task Configurations (Mock & Real configs)
- Added specific configuration parameters for `copy` and `abc` tasks:
  - **copy task**:
    - Mock config: `seeds=[1], epochs=2, train_len=5, test_len=10, vocab_size=4, num_samples=5`.
    - Real config: `seeds=list(range(1, 11)), epochs=80, train_len=20, test_len=100, vocab_size=10, num_samples=100`.
  - **abc task**:
    - Mock config: `seeds=[1], epochs=2, train_len=5, test_len=10, vocab_size=4, num_samples=5`.
    - Real config: `seeds=list(range(1, 11)), epochs=80, train_len=20, test_len=100, vocab_size=4, num_samples=100`.

### 4. Dynamic Model Selection
- Handled models to run dynamically depending on the task:
  - If task is `copy` or `abc`, `--model all` resolves to evaluating `StackRNN` and `DualStackRNN`.
- Passed proper `vocab_size` to all models during instantiation.

### 5. Task-specific Dataset Generation & Masking Settings
- Generated datasets inside the seed loop using `generate_copy_task` and `generate_abc_task` with length generalization enforced (`train_len` vs `test_len`).
- Set `ignore_index` to `None` for copy task, and `3` for abc task.

### 6. Ignore Index Handling in Train & Evaluation
- Updated `train_model` to pass `ignore_index` to `nn.CrossEntropyLoss`.
- Updated `evaluate_model_accs` to accept `ignore_index` and correctly mask token and sequence accuracy calculations. Specifically:
  - Sequence accuracy: `((preds == Y) | ~mask).all(dim=-1).float().mean().item()` where `mask = (Y != ignore_index)`.

### 7. Metrics CSV Generation
- Formatted metrics (`model,accuracy,token_accuracy,sequence_accuracy,sparsity`) precisely to 4 decimal places and saved them to `results_table.csv` under the specified output directory.
