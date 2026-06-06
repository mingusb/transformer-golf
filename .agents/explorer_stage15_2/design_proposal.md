# Design Proposal: Stage 15 (Evaluation & Integration)

This document presents the detailed design and implementation specifications for integrating the context-sensitive task evaluations (`copy` and `abc` tasks) and the `DualStackRNN` model into the experimental framework in `src/scripts/run_experiments.py`.

---

## 1. DualStackRNN Import and Availability Flag

To dynamically support the `DualStackRNN` model when it is available in the codebase, we use a try-except block at the module level of `src/scripts/run_experiments.py`. This mimics the current pattern used for `StackRNN` and `LiquidStateMachine`.

### Specification
```python
try:
    from src.models.universal_rnn import DualStackRNN
    HAS_DUAL_STACK_RNN = True
except ImportError:
    HAS_DUAL_STACK_RNN = False
```

This guarantees that:
- If the file `src/models/universal_rnn.py` contains the class `DualStackRNN`, `HAS_DUAL_STACK_RNN` will be `True`.
- If the class is missing or cannot be imported, the script continues executing without crashing, setting the flag to `False`.

---

## 2. CLI Argparse and Model Routing Configuration

The argparse configurations for `--task` and `--model` must be expanded to accommodate the new evaluation targets.

### Specification
1. Update `parser.add_argument` definitions inside `main()`:
```python
    parser.add_argument("--task", type=str, choices=["alternating", "nesting", "copy", "abc"], default="alternating")
    parser.add_argument("--model", type=str, default="all", choices=["all", "ssm", "attention", "conv1d", "markov", "stack_rnn", "lsm", "dual_stack_rnn"])
```

2. Update the dynamic routing logic to resolve available models:
```python
    if args.task in ["alternating", "copy", "abc"]:
        # Determine which models to run
        if args.model == "all":
            models_to_run = ["SSM", "Attention", "Conv1D", "MarkovChain"]
            if HAS_LSM:
                models_to_run.append("LSM")
            if HAS_STACK_RNN:
                models_to_run.append("StackRNN")
            if HAS_DUAL_STACK_RNN:
                models_to_run.append("DualStackRNN")
        else:
            model_map = {
                "ssm": ["SSM"],
                "attention": ["Attention"],
                "conv1d": ["Conv1D"],
                "markov": ["MarkovChain"],
                "stack_rnn": ["StackRNN"] if HAS_STACK_RNN else [],
                "lsm": ["LSM"] if HAS_LSM else [],
                "dual_stack_rnn": ["DualStackRNN"] if HAS_DUAL_STACK_RNN else []
            }
            models_to_run = model_map.get(args.model, [])
            if not models_to_run:
                print(f"Requested model {args.model} is not available/implemented.")
                return 1
```

---

## 3. Task Hyperparameters, Vocabulary, and Padding Verification

The table below details the proposed hyperparameters for both tasks under the `mock` and `real_config` modes.

### Training Configuration Specifications

| Task | Configuration | Seeds | Epochs | Learning Rate | Train Length ($L$ or $n$) | Test Length ($L$ or $n$) | Vocab Size | Samples |
|---|---|---|---|---|---|---|---|---|
| **Copy** | Mock | `[1]` | 2 | 0.03 | 3 | 6 | 4 | 5 |
| **Copy** | Real | `1..10` | 80 | 0.03 | 10 | 20 | 10 | 100 |
| **ABC** | Mock | `[1]` | 2 | 0.03 | 2 | 4 | 4 (Fixed) | 5 |
| **ABC** | Real | `1..10` | 80 | 0.03 | 10 | 20 | 4 (Fixed) | 100 |

### Task-Specific Verification

#### 1. Copy Task
- **Inputs**: Generated using `generate_copy_task(num_samples, length, vocab_size)`. Sequence structure is `w # w` where `w` consists of tokens in `[0, vocab_size - 2]`, and `#` is `vocab_size - 1`. The total length is $2L + 1$.
- **Targets**: Left-shifted inputs, with the final step set to `0`.
- **Vocabulary Size**: The parameter `vocab_size` governs both model output channels and dataset generation.

#### 2. ABC Task
- **Inputs**: Generated using `generate_abc_task(num_samples, n_max, n)`. Sequence structure is $a^n b^n c^n$. 
- **Vocabulary Size**: The vocabulary is **exactly 4** (represented by tokens `0` for $a$, `1` for $b$, `2` for $c$, and `3` for padding/EOS). Therefore, the model's `vocab_size` parameter **must be set to 4**.
- **Padding Behavior**: 
  - When calling with a fixed `n`, the input sequences have length `3 * n` and contain only `[0, 1, 2]`. No padding is present in the inputs. The targets are left-shifted inputs with the last element set to the EOS token `3`.
  - When calling with `n = None`, sequences are right-padded to `3 * n_max` using token `3`. Targets are also right-padded.
  - To test length generalization reliably and avoid padding bias during evaluation, the recommended design is to use a specified `n` for training (e.g. `n = train_len`) and a larger specified `n` for testing (e.g. `n = test_len`).

---

## 4. Data Generator Calls

The calls to the generators inside the main seed loop in `run_experiments.py` should be configured as follows:

```python
# For Copy Task
if args.task == "copy":
    X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
    X_test, Y_test = generate_copy_task(num_samples, test_len, vocab_size)

# For ABC Task
elif args.task == "abc":
    # Using fixed n for clean length generalization testing
    X_train, Y_train = generate_abc_task(num_samples, n_max=train_len, n=train_len)
    X_test, Y_test = generate_abc_task(num_samples, n_max=test_len, n=test_len)
```

---

## 5. Evaluation Loop and Metric Computation

### Output Structure Compatibility
The forward method returns:
- `StackRNN`: `(logits, stack_states)` where `stack_states` is a single tensor.
- `DualStackRNN`: `(logits, (stack1_states, stack2_states))` where the second item is a tuple of two tensors.
- Baselines/SSM: `(logits, None)`.

Both training and evaluation loops unpack this using:
```python
logits, _ = model(X)
```
Since the second output value is bound to the discard variable `_`, Python's unpacking handles all three structures seamlessly without raising unpacking errors.

### Metric Calculation (Token & Sequence Accuracies)
To handle tasks containing padding tokens (e.g., the EOS/pad token `3` in `abc`), accuracy calculations should support masking. This ensures metrics measure performance on actual content rather than inflating scores by predicting padding characters.

```python
def evaluate_model_accs(model, X, Y, pad_token=None):
    model.eval()
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        
        if pad_token is not None:
            mask = (Y != pad_token)
            if mask.sum() > 0:
                token_acc = (preds[mask] == Y[mask]).float().mean().item()
                # A sequence is correct if and only if all non-pad tokens match targets
                seq_correct = ((preds == Y) | ~mask).all(dim=-1).float()
                seq_acc = seq_correct.mean().item()
            else:
                token_acc = 0.0
                seq_acc = 0.0
        else:
            token_acc = (preds == Y).float().mean().item()
            seq_acc = (preds == Y).all(dim=-1).float().mean().item()
            
    return token_acc, seq_acc
```

*Note: For the `abc` task, `pad_token` should be set to `3`. For the `copy` task, no masking is required since there are no padding tokens inside the inputs.*

### Sparsity Metric
- For **SSM** models with L0 pruning, sparsity is retrieved from the Pareto frontier:
  `sparsity = masked_ssm.pareto_frontier[-1].sparsity`
- For all other models (including `StackRNN` and `DualStackRNN`), sparsity is hardcoded to `0.0`.

---

## 6. Saving Results to CSV

All task evaluation outputs must be formatted identically to maintain compatibility with downstream plotting and parsing utilities.

### CSV Layout Specifications
The summary results table must be saved to `output_dir/results_table.csv` with the following columns:
`model,accuracy,token_accuracy,sequence_accuracy,sparsity`

Where:
- `accuracy` and `token_accuracy` are both populated with the mean token accuracy.
- `sequence_accuracy` is populated with the mean sequence accuracy.
- `sparsity` is populated with the mean L0 sparsity (or `0.0` for non-SSM models).

### Saving Routine
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

---

## 7. Device Placement and Mitigation of Edge Cases

### The L0 Masking Device Placement Mismatch (CRITICAL)
In `src/models/sparsity.py` (lines 97-100), the initialization in `apply_l0_mask` creates validation tensors on the default device (CPU) without passing a `device` parameter:
```python
val_x = torch.randint(0, vocab_size, (10, 5))
val_y = torch.randint(0, vocab_size, (10, 5))
with torch.no_grad():
    out = model(val_x)
```
If a model has been transferred to a GPU (`cuda`) prior to applying the L0 mask, `out = model(val_x)` will trigger a device mismatch crash:
`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

### Recommended Mitigation
Because the models (`d_model=8`, `state_dim=16`) and batch sizes (`num_samples=100`) are tiny, training is extremely fast. CPU execution takes only a few seconds per seed and avoids all device mismatch issues.
Thus:
1. **Default to CPU**: Execute all training and validation runs on CPU. This completely avoids the device mismatch issue in `apply_l0_mask` and is highly efficient.
2. **GPU Workaround (if CUDA is mandated)**:
   - Call `apply_l0_mask` on the model while it is still on the CPU.
   - After the mask is applied, move the model to CUDA: `model = model.to(device)`.
   - Set the validation data attributes using GPU tensors to prevent crashes in subsequent pruning steps:
     ```python
     model.val_data = (val_x.to(device), val_y.to(device))
     ```
   - Ensure all input datasets are placed on `device` before passing them to the training or evaluation functions:
     ```python
     X_train, Y_train = X_train.to(device), Y_train.to(device)
     X_test, Y_test = X_test.to(device), Y_test.to(device)
     ```
