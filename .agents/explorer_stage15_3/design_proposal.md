# Design Proposal: Stage 15 (Evaluation & Integration)

This document presents the detailed design for implementing **Stage 15 (Evaluation & Integration)** in the project. The primary focus of this stage is to integrate the Copy Task and ABC Task (Multiple Counting Task) into the experiment script `src/scripts/run_experiments.py`, import and support the `DualStackRNN` model, and compare it against the `StackRNN` baseline under length generalization settings.

---

## 1. Importing `DualStackRNN` and Defining `HAS_DUAL_STACK_RNN`

To support the Dual-Stack Augmented Recurrent Neural Network (`DualStackRNN`) dynamically, we follow the established pattern used for `StackRNN` and `LiquidStateMachine` (LSM). 

### Proposed Code Change
At the top of `src/scripts/run_experiments.py` (around line 23-35), add the following import block:

```python
# Import StackRNN and LSM
try:
    from src.models.stack_rnn import StackRNN
    HAS_STACK_RNN = True
except ImportError:
    HAS_STACK_RNN = False

try:
    from src.models.lsm import LiquidStateMachine
    HAS_LSM = True
except ImportError:
    HAS_LSM = False

# Import DualStackRNN (Stage 15 Integration)
try:
    from src.models.universal_rnn import DualStackRNN
    HAS_DUAL_STACK_RNN = True
except ImportError:
    HAS_DUAL_STACK_RNN = False
```

This ensures that the script remains executable even if the `universal_rnn.py` module is not present or if there are missing dependencies.

---

## 2. Updating Argparse Choices for `--task` and `--model`

The argparse argument definitions in `main()` must be updated to support the new tasks (`"copy"` and `"abc"`) and the new model (`"dual_stack_rnn"`).

### Proposed Code Change
Modify the argument parser setup in `src/scripts/run_experiments.py` (around lines 72-77) as follows:

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

Additionally, update the model resolution logic when `args.model` is `"all"` or specific models are requested for the new tasks:
- If `--task` is `"copy"` or `"abc"` and `--model` is `"all"`, the default comparison models should be `["StackRNN", "DualStackRNN"]` (filtered by their import flags `HAS_STACK_RNN` and `HAS_DUAL_STACK_RNN`).
- If a specific model is requested, map it correctly (e.g., `"dual_stack_rnn"` maps to `["DualStackRNN"]`).

---

## 3. Training Configurations for Copy and ABC Tasks

To test length generalization, the model is trained on shorter sequence lengths and evaluated on significantly longer sequences. 

### Configuration Parameters

| Task | Configuration | Seeds | Epochs | Learning Rate | Train Length (`train_len`) | Test Length (`test_len`) | Vocab Size | Samples | Stack Depth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Copy** | Mock | `[1]` | 2 | 0.03 | 5 (seq len: 11) | 10 (seq len: 21) | 4 | 5 | 15 |
| **Copy** | Real | `1..10` | 80 | 0.03 | 20 (seq len: 41) | 100 (seq len: 201) | 10 | 100 | 150 |
| **ABC** | Mock | `[1]` | 2 | 0.03 | 5 (seq len: 15) | 10 (seq len: 30) | 4 | 5 | 15 |
| **ABC** | Real | `1..10` | 80 | 0.03 | 20 (seq len: 60) | 100 (seq len: 300) | 4 | 100 | 150 |

### ABC Task Vocabulary and Padding Details

1. **Vocabulary Size**: The vocabulary size is strictly **4**. 
   - `0` represents character `'a'`
   - `1` represents character `'b'`
   - `2` represents character `'c'`
   - `3` represents the padding/EOS token
2. **Padding and Sequence Length Behavior**:
   - **Fixed Length (`n` specified)**: If `n = count` is specified, the sequence generated has exactly length `3 * n` (no padding in inputs). The targets are left-shifted inputs with the last element set to the padding/EOS token `3`.
   - **Variable Length (`n = None`)**: If `n` is `None`, each sequence has its count randomly sampled from `[1, n_max]`, and the input is right-padded with the padding token `3` to reach a length of `3 * n_max`.
   - **Design Decision**: For clean length generalization testing, the fixed length option (`n=train_len` for training, and `n=test_len` for testing) is highly recommended. It avoids masking overhead during training and tests the counting capacity directly. If variable length (`n=None`) is selected, the loss function and accuracy metrics **must** ignore index `3` (padding) to prevent artificial inflation of accuracy.

---

## 4. Setting up Data Loader/Generator Calls

The task generators are imported from `src.data.context_sensitive`. 

### Generator Signatures
- `generate_copy_task(num_samples: int, length: int, vocab_size: int)`
- `generate_abc_task(num_samples: int, n_max: int, n: int = None)`

### Implementation Calls

For the **Copy Task**:
```python
# Train generation
X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
# Test generation
X_test, Y_test = generate_copy_task(num_samples, test_len, vocab_size)
```

For the **ABC Task** (using fixed count `n` for strict length generalization):
```python
# Train generation: sequences of length 3 * train_len
X_train, Y_train = generate_abc_task(num_samples, n_max=train_len, n=train_len)
# Test generation: sequences of length 3 * test_len
X_test, Y_test = generate_abc_task(num_samples, n_max=test_len, n=test_len)
```

If variable length `n` is used:
```python
# Train generation: padded to 3 * train_len
X_train, Y_train = generate_abc_task(num_samples, n_max=train_len, n=None)
# Test generation: padded to 3 * test_len
X_test, Y_test = generate_abc_task(num_samples, n_max=test_len, n=None)
```

---

## 5. Evaluation Loop, Output Structure Handling, and Metrics

### Output Structures
- `StackRNN` returns a tuple `(logits, stack_states)` where `stack_states` is a single tensor.
- `DualStackRNN` returns a tuple `(logits, (stack1_states, stack2_states))` where the second element is a tuple of two tensors.
- Other baselines return `(logits, hidden_state)`.

Because all models return a tuple where the first element is the `logits` tensor, we can use the generic extraction:
```python
logits, _ = model(X)
```
This is fully compatible with all models.

### Metric Calculations
To support tasks with padding (such as variable length ABC task), the training loss and accuracy calculations must support an optional `ignore_index`.

#### Custom Training Function
```python
def train_model(model, X_train, Y_train, epochs=50, lr=0.03, ignore_index=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        loss.backward()
        optimizer.step()
```

#### Custom Accuracy Evaluation
```python
def evaluate_model_accs(model, X, Y, ignore_index=None):
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        if ignore_index is not None:
            mask = (Y != ignore_index)
            if mask.sum() == 0:
                token_acc = 1.0
                seq_acc = 1.0
            else:
                token_acc = (preds[mask] == Y[mask]).float().mean().item()
                # Compute sequence accuracy only on non-ignored elements
                correct_seqs = [
                    (preds[i][mask[i]] == Y[i][mask[i]]).all().item() 
                    for i in range(Y.size(0))
                ]
                seq_acc = sum(correct_seqs) / len(correct_seqs)
        else:
            token_acc = (preds == Y).float().mean().item()
            seq_acc = (preds == Y).all(dim=-1).float().mean().item()
    return token_acc, seq_acc
```

#### Sparsity Metric
Neither `StackRNN` nor `DualStackRNN` are structurally pruned using L0 regularization in the context-sensitive baselines. Therefore, we report **0.0** sparsity for these models, storing it in the results dictionary to maintain layout compatibility.

---

## 6. Formatting and Saving Results

The comparison results must be saved in `output_dir/results_table.csv` using the strict schema required by downstream parser/tests:

### CSV Schema
`model,accuracy,token_accuracy,sequence_accuracy,sparsity`

- **Important**: Both the `accuracy` and `token_accuracy` columns must be populated with the mean token accuracy.
- All numeric values should be formatted to **4 decimal places** (`.4f`).

### Python Writing Implementation
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

## 7. Device Placement and Edge Cases (CPU vs. CUDA)

### 1. Pruning Verification Conflict
A critical edge case exists in `src/models/sparsity.py` inside `apply_l0_mask()` and `l0_pruning_step()`:
- These helper functions create validation tensors `val_x` and `val_y` on the CPU default device:
  ```python
  val_x = torch.randint(0, vocab_size, (10, 5))
  ```
- If the model parameters have been moved to GPU (via `model.to('cuda')`), calling `model(val_x)` will raise a device mismatch `RuntimeError` (e.g. `Expected all tensors to be on the same device...`).
- **Recommendation**: For these small-scale experiments, run all models and training **solely on CPU**. CPU execution is extremely fast (under 1 second per seed) and entirely bypasses device mismatch errors in the pruning module.

### 2. Stack Representations Device Alignment
Within `StackRNN.forward` and `DualStackRNN.forward`, the internal stack states `S`, `S1`, and `S2` are initialized using the device of the input tensor `x` (i.e. `device=x.device`). 
- This means if inputs are moved to a device (e.g., CPU or CUDA), the stacks will automatically reside on the correct device.
- Therefore, if GPU execution is ever required, moving both the model and the input tensors to the target device is sufficient:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  X_train = X_train.to(device)
  Y_train = Y_train.to(device)
  ```
  *Note: To prevent the pruning verification conflict mentioned above, ensure L0 pruning checks are executed only on CPU.*

---

## 8. Proposed Execution Blueprint for `run_experiments.py`

Below is the structured code snippet that the implementing agent should inject into `src/scripts/run_experiments.py`:

```python
    elif args.task in ["copy", "abc"]:
        print(f"Running {args.task.capitalize()} Task experiment...")
        
        # 1. Determine configuration hyperparameters
        is_abc = (args.task == "abc")
        if args.config == "mock":
            seeds = [1]
            epochs = 2
            train_len = 5
            test_len = 10
            vocab_size = 4
            num_samples = 5
        else:
            seeds = list(range(1, 11))
            epochs = 80
            train_len = 20
            test_len = 100
            vocab_size = 4 if is_abc else 10
            num_samples = 100

        # Define stack depth: must be large enough to hold test sequence
        stack_depth = test_len + 10

        # Determine which models to run
        if args.model == "all":
            models_to_run = []
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

        results = {m: {"token_accs": [], "seq_accs": [], "sparsities": []} for m in models_to_run}

        for seed in seeds:
            print(f"\n--- Running Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Generate task data
            if is_abc:
                # Fixed count 'n' used to enforce strict length generalization
                X_train, Y_train = generate_abc_task(num_samples, n_max=train_len, n=train_len)
                X_test, Y_test = generate_abc_task(num_samples, n_max=test_len, n=test_len)
                # Pad/EOS index is 3
                ignore_index = 3 
            else:
                X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
                X_test, Y_test = generate_copy_task(num_samples, test_len, vocab_size)
                ignore_index = None

            # Instantiate models
            models = {}
            if "SSM" in models_to_run:
                models["SSM"] = RecurrentSSM(vocab_size=vocab_size, d_model=8, state_dim=16)
            if "Attention" in models_to_run:
                models["Attention"] = CausalAttentionModel(vocab_size=vocab_size, d_model=8, state_dim=16)
            if "Conv1D" in models_to_run:
                models["Conv1D"] = Conv1DModel(vocab_size=vocab_size, d_model=8, state_dim=16)
            if "MarkovChain" in models_to_run:
                models["MarkovChain"] = MarkovChainModel(vocab_size=vocab_size, d_model=8, state_dim=16)
            if "StackRNN" in models_to_run:
                models["StackRNN"] = StackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=stack_depth)
            if "DualStackRNN" in models_to_run:
                models["DualStackRNN"] = DualStackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=stack_depth)
            if "LSM" in models_to_run:
                models["LSM"] = LiquidStateMachine(input_size=vocab_size, reservoir_size=50, output_size=vocab_size, spectral_radius=0.99, sparsity=0.1)

            # Train models
            for m_name, model in models.items():
                print(f"Training {m_name} on {args.task} task...")
                train_model(model, X_train, Y_train, epochs=epochs, lr=0.03, ignore_index=ignore_index)

            # Evaluate models on test length (length generalization)
            for m_name, model in models.items():
                token_acc, seq_acc = evaluate_model_accs(model, X_test, Y_test, ignore_index=ignore_index)
                results[m_name]["token_accs"].append(token_acc)
                results[m_name]["seq_accs"].append(seq_acc)
                results[m_name]["sparsities"].append(0.0)

        # Print final results summary
        print(f"\n================ FINAL {args.task.upper()} RESULTS ================")
        for name in models_to_run:
            mean_token = np.mean(results[name]["token_accs"])
            mean_seq = np.mean(results[name]["seq_accs"])
            print(f"{name}:")
            print(f"  Token Acc: {mean_token:.4f}")
            print(f"  Seq Acc:   {mean_seq:.4f}")

        # T-Test Comparison (DualStackRNN vs StackRNN)
        if "DualStackRNN" in models_to_run and "StackRNN" in models_to_run and len(seeds) > 1:
            print("\n--- Statistical Significance (DualStackRNN vs StackRNN) ---")
            p_t_token = run_t_test(results["DualStackRNN"]["token_accs"], results["StackRNN"]["token_accs"], method="welch")
            p_mw_token = run_t_test(results["DualStackRNN"]["token_accs"], results["StackRNN"]["token_accs"], method="mann_whitney")
            p_t_seq = run_t_test(results["DualStackRNN"]["seq_accs"], results["StackRNN"]["seq_accs"], method="welch")
            p_mw_seq = run_t_test(results["DualStackRNN"]["seq_accs"], results["StackRNN"]["seq_accs"], method="mann_whitney")
            
            print("DualStackRNN vs StackRNN:")
            print(f"  Token Acc: Welch p-val = {p_t_token:.4f}, Mann-Whitney p-val = {p_mw_token:.4f}")
            print(f"  Seq Acc:   Welch p-val = {p_t_seq:.4f}, Mann-Whitney p-val = {p_mw_seq:.4f}")

        # Write the results table to results_table.csv
        table_path = os.path.join(args.output_dir, "results_table.csv")
        with open(table_path, "w") as f:
            f.write("model,accuracy,token_accuracy,sequence_accuracy,sparsity\n")
            for name in models_to_run:
                mean_token = np.mean(results[name]["token_accs"])
                mean_seq = np.mean(results[name]["seq_accs"])
                mean_sparsity = np.mean(results[name]["sparsities"])
                f.write(f"{name},{mean_token:.4f},{mean_token:.4f},{mean_seq:.4f},{mean_sparsity:.4f}\n")
        
        print(f"Results saved to {args.output_dir}")
        return 0
```
