# Design Proposal: Stage 15 (Evaluation & Integration)

This document outlines the detailed design proposal for implementing Stage 15 of the project in `src/scripts/run_experiments.py`.

---

## Section 1: Importing `DualStackRNN` and Defining `HAS_DUAL_STACK_RNN`

### Objective
Integrate the `DualStackRNN` model class defined in `src/models/universal_rnn.py` into the evaluation script, maintaining robust handling of optional model availability.

### Current Implementation (for StackRNN)
```python
try:
    from src.models.stack_rnn import StackRNN
    HAS_STACK_RNN = True
except ImportError:
    HAS_STACK_RNN = False
```

### Proposed Update
```python
try:
    from src.models.stack_rnn import StackRNN
    HAS_STACK_RNN = True
except ImportError:
    HAS_STACK_RNN = False

try:
    from src.models.universal_rnn import DualStackRNN
    HAS_DUAL_STACK_RNN = True
except ImportError:
    HAS_DUAL_STACK_RNN = False
```

---

## Section 2: Command-Line Interface (Argparse) Updates

### Objective
Expose the new context-sensitive tasks (`copy` and `abc`) and the new model (`dual_stack_rnn`) in the CLI arguments to allow target-specific experiments.

### Proposed Update
In the `main()` function:
```python
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="real_config")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["alternating", "nesting", "copy", "abc"], 
        default="alternating"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="all", 
        choices=["all", "ssm", "attention", "conv1d", "markov", "stack_rnn", "lsm", "dual_stack_rnn"]
    )
```

Additionally, update the `models_to_run` selection mapping logic:
```python
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
        models_to_run = model_map.get(args.model, ["SSM", "Attention", "Conv1D", "MarkovChain"])
        if not models_to_run:
            print(f"Requested model {args.model} is not available/implemented.")
            return 1
```

---

## Section 3: Training & Data Configurations

We define configurations for `copy` and `abc` tasks, splitting them into `mock` (for rapid testing/integration tests) and `real` (for full training and profiling).

### 1. Copy Task Configuration
- **Vocabulary Size (`vocab_size`)**: `4` (both mock and real). This includes tokens `0, 1, 2` for sequence items, and token `3` for the `#` delimiter.
- **Learning Rate (`lr`)**: `0.03`
- **Mock Config**:
  - `epochs = 2`
  - `num_samples = 5`
  - `train_len = 5` (sub-sequence `w` length; total sequence length = `11`)
  - `test_len = 10` (total sequence length = `21`)
  - `seeds = [1]`
- **Real Config**:
  - `epochs = 80`
  - `num_samples = 100`
  - `train_len = 20` (total sequence length = `41`)
  - `test_len = 100` (total sequence length = `201`)
  - `seeds = list(range(1, 11))`

### 2. ABC Task Configuration
- **Vocabulary Size (`vocab_size`)**: `4` (tokens: `0` for 'a', `1` for 'b', `2` for 'c', and `3` for padding/EOS).
- **Learning Rate (`lr`)**: `0.03`
- **Mock Config**:
  - `epochs = 2`
  - `num_samples = 5`
  - `train_len = 5` (maximum count `n_max = 5`; sequence length = `15`)
  - `test_len = 10` (maximum count `n_max = 10`; sequence length = `30`)
  - `seeds = [1]`
- **Real Config**:
  - `epochs = 80`
  - `num_samples = 100`
  - `train_len = 20` (maximum count `n_max = 20`; sequence length = `60`)
  - `test_len = 100` (maximum count `n_max = 100`; sequence length = `300`)
  - `seeds = list(range(1, 11))`

### 3. ABC Task Vocabulary Size and Padding Verification
- **Vocabulary Size Verification**: The generator `generate_abc_task` outputs tokens in `{0, 1, 2, 3}`. The target sequences use token `3` as the shifted ending token. Therefore, models must be initialized with `vocab_size = 4`.
- **Padding Behavior**: When generating with `n=None` (for length and count diversity), sequences are padded to `3 * n_max` using token `3`. Targets are left-shifted and padded with token `3` at the end. Without mask handling, this padding token would artificially inflate token accuracy. To prevent this, evaluation and loss calculations should mask out token `3`.

---

## Section 4: Data Generator Setup

We import the generators dynamically and define the dataset generation lambda function depending on the chosen task.

```python
    # Determine vocab size, pad token, and generator based on task
    if args.task == "alternating":
        vocab_size = 2
        generate_fn = lambda length, samples: generate_alternating(length, samples)
        pad_token = None
    elif args.task == "copy":
        vocab_size = 4
        from src.data.context_sensitive import generate_copy_task
        generate_fn = lambda length, samples: generate_copy_task(num_samples=samples, length=length, vocab_size=vocab_size)
        pad_token = None
    elif args.task == "abc":
        vocab_size = 4
        from src.data.context_sensitive import generate_abc_task
        generate_fn = lambda length, samples: generate_abc_task(num_samples=samples, n_max=length, n=None)
        pad_token = 3
    else:
        # Nesting task is handled separately
        pass
```

During the seed training/evaluation loops, the generators are invoked as:
```python
        X_train, Y_train = generate_fn(train_len, num_samples)
        X_test, Y_test = generate_fn(test_len, num_samples)
```

---

## Section 5: Evaluation Loop, Model Output Unpacking, and Metrics

### 1. Model Output Unpacking
PyTorch models return tuples of `(logits, state)`. 
- `StackRNN` returns: `(logits_tensor, stack_states_tensor)` where the second element is a tensor.
- `DualStackRNN` returns: `(logits_tensor, (stack1_states_tensor, stack2_states_tensor))` where the second element is a tuple of two tensors.

Since both return the main prediction `logits` as the first element, Python's sequence unpacking `logits, _ = model(X)` works seamlessly for both models without explicit type checking.

### 2. Loss Masking in Training
To prevent padding tokens from influencing gradients:
```python
def train_model(model, X_train, Y_train, epochs=50, lr=0.03, pad_token=None, device='cpu'):
    model = model.to(device)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if pad_token is not None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        loss.backward()
        optimizer.step()
```

### 3. Metric Computations in Evaluation
For tasks with padding (`abc`), token accuracy and sequence accuracy should exclude padding tokens:
```python
def evaluate_model_accs(model, X, Y, pad_token=None, device='cpu'):
    model = model.to(device)
    X = X.to(device)
    Y = Y.to(device)
    
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        if pad_token is not None:
            mask = (Y != pad_token)
            num_valid = mask.sum().item()
            if num_valid > 0:
                token_acc = ((preds == Y) & mask).float().sum().item() / num_valid
            else:
                token_acc = 1.0
            
            # Sequence accuracy: all non-pad tokens in a sequence must be correctly predicted
            correct = (preds == Y) | ~mask
            seq_acc = correct.all(dim=-1).float().mean().item()
        else:
            token_acc = (preds == Y).float().mean().item()
            seq_acc = (preds == Y).all(dim=-1).float().mean().item()
            
    return token_acc, seq_acc
```

---

## Section 6: Formatting and Saving Results

The results must be saved to `output_dir/results_table.csv` using the existing format to maintain compatibility with downstream analysis tools:

```csv
model,accuracy,token_accuracy,sequence_accuracy,sparsity
```

### Implementation Detail
```python
    table_path = os.path.join(args.output_dir, "results_table.csv")
    with open(table_path, "w") as f:
        f.write("model,accuracy,token_accuracy,sequence_accuracy,sparsity\n")
        for name in models_to_run:
            mean_token = np.mean(results[name]["token_accs"])
            mean_seq = np.mean(results[name]["seq_accs"])
            mean_sparsity = np.mean(results[name]["sparsities"])
            # 'accuracy' column mirrors token_accuracy
            f.write(f"{name},{mean_token:.4f},{mean_token:.4f},{mean_seq:.4f},{mean_sparsity:.4f}\n")
```

---

## Section 7: Device Placement & Edge Cases (CPU vs CUDA)

### 1. Robust Device Detection
Determine the default device at the beginning of the script:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. Model & Input Alignment
All instantiated models and datasets must be transferred to the target `device` prior to training/evaluation. This avoids `RuntimeError` mismatch exceptions.

### 3. L0 Masking / Pruning Dummy Tensors (Critical Edge Case)
Inside `src/models/sparsity.py`, `apply_l0_mask` and `l0_pruning_step` create dummy verification tensors on the default device (which is CPU):
```python
val_x = torch.randint(0, vocab_size, (10, 5))
```
If the model has already been moved to the GPU, `model(val_x)` will trigger a device mismatch error.

**Workaround Design**:
1. Temporarily move the model to the CPU before applying the mask.
2. Apply the mask (`apply_l0_mask`).
3. Move the masked model back to the target GPU device.
4. Set the `model.val_data` property to a tuple of target device-placed validation tensors. This ensures that `l0_pruning_step` uses the GPU-resident tensors instead of generating CPU dummy ones.

```python
        # Apply L0 pruning to SSM if present
        if "SSM" in models:
            print("Applying L0 pruning to SSM...")
            # Workaround for sparsity.py device mismatch
            models["SSM"] = models["SSM"].to("cpu")
            masked_ssm = apply_l0_mask(models["SSM"])
            masked_ssm = masked_ssm.to(device)
            # Pre-populate val_data on target device to prevent CPU generation in l0_pruning_step
            masked_ssm.val_data = (X_test.to(device), Y_test.to(device))
            l0_pruning_step(masked_ssm, temperature=0.1, target_sparsity=0.2)
            eval_ssm = masked_ssm
            ssm_sparsity = masked_ssm.pareto_frontier[-1].sparsity
        else:
            ssm_sparsity = 0.0
```
This workaround completely eliminates device mismatch errors under CUDA without requiring changes to the core `sparsity.py` implementation.
