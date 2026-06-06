# Handoff Report: Stage 15 (Evaluation & Integration) Explorer 3

## 1. Observation

- **Imports and Flag**:
  In `src/scripts/run_experiments.py`, models like `StackRNN` (lines 23-28) and `LiquidStateMachine` (lines 29-33) are conditionally imported using `try-except` blocks.
  ```python
  try:
      from src.models.stack_rnn import StackRNN
      HAS_STACK_RNN = True
  except ImportError:
      HAS_STACK_RNN = False
  ```
- **Argparse Config**:
  In `src/scripts/run_experiments.py` (lines 75-76), argument choices for `--task` and `--model` are defined as:
  ```python
  parser.add_argument("--task", type=str, choices=["alternating", "nesting"], default="alternating")
  parser.add_argument("--model", type=str, default="all", choices=["all", "ssm", "attention", "conv1d", "markov", "stack_rnn", "lsm"])
  ```
- **Forward Output Structures**:
  - `src/models/stack_rnn.py` (lines 27-38) defines the return type of `StackRNN.forward` as:
    ```python
    Returns:
        logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size)
        stack_states (torch.Tensor): Soft stack states of shape (batch_size, seq_len, stack_depth, stack_width)
    ```
  - `src/models/universal_rnn.py` (lines 32-44) defines the return type of `DualStackRNN.forward` as:
    ```python
    Returns:
        logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size)
        stack_states (Tuple[torch.Tensor, torch.Tensor]): Tuple of stack 1 and stack 2 states...
    ```
- **Task Generators**:
  In `src/data/context_sensitive.py` (lines 4-8):
  ```python
  def generate_copy_task(
      num_samples: int,
      length: int,
      vocab_size: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
  ```
  And lines 58-62:
  ```python
  def generate_abc_task(
      num_samples: int,
      n_max: int,
      n: int = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
  ```
- **ABC Task Vocabulary and Padding**:
  In `src/data/context_sensitive.py` (lines 106-116):
  - Tokens used in `inputs` for `abc` task: `0` for 'a', `1` for 'b', `2` for 'c', and `3` for padding.
  - The padding/EOS token `3` is appended to targets:
    ```python
    padding = torch.full((num_samples, 1), 3, dtype=torch.long)
    targets = torch.cat([inputs[:, 1:], padding], dim=1)
    ```
- **Device Placement Edge Case**:
  In `src/models/sparsity.py` (lines 96-103):
  ```python
  vocab_size = getattr(model, "vocab_size", 2)
  val_x = torch.randint(0, vocab_size, (10, 5))
  val_y = torch.randint(0, vocab_size, (10, 5))
  with torch.no_grad():
      out = model(val_x)
  ```
  Here `val_x` and `val_y` are instantiated on CPU by default. If `model` parameters are moved to GPU, calling `model(val_x)` triggers a device mismatch `RuntimeError`.

---

## 2. Logic Chain

1. **Imports & Flag**: Conditional importing via `try-except` (similar to lines 23-28 in `run_experiments.py`) will allow the script to import `DualStackRNN` safely and define `HAS_DUAL_STACK_RNN` to toggle its inclusion in experimental runs.
2. **Argparse choices**: Expanding `--task` to `["alternating", "nesting", "copy", "abc"]` and `--model` to include `"dual_stack_rnn"` permits users to configure the execution options via command-line arguments.
3. **Model outputs**: Both `StackRNN.forward` and `DualStackRNN.forward` return a tuple where the first element is the `logits` tensor. Thus, the generic assignment `logits, _ = model(x)` is syntactically compatible with both models.
4. **Task configurations**: 
   - For `copy` task: `vocab_size` must support characters in `w` and the delimiter. Setting it to 10 for the real configuration is safe.
   - For `abc` task: the vocabulary is strictly size 4 (0: 'a', 1: 'b', 2: 'c', 3: pad/EOS). 
5. **Padding & evaluation**: Using fixed `n` (`n = train_len` / `n = test_len`) avoids padding tokens in sequence inputs, preventing target sequences from being dominated by pad predictions. If variable-length `n` is used, the training loss and accuracy evaluation must explicitly mask out the pad index `3`.
6. **Device placement**: In `sparsity.py`, since dummy validation tensors are hardcoded on CPU, any verification check (e.g. `apply_l0_mask` or `l0_pruning_step`) will crash if the model is moved to GPU. Keeping all experiments on CPU is the recommended path for CPU/CUDA compatibility.

---

## 3. Caveats

- We assumed that structural pruning (L0 regularization) is not required for `StackRNN` or `DualStackRNN` under the Copy or ABC tasks; hence their sparsity is reported as `0.0`.
- If GPU training is explicitly requested by the user, the pruning verification functions in `src/models/sparsity.py` must be patched or run on CPU before transferring the model to the GPU.

---

## 4. Conclusion

The implementation of Stage 15 requires:
1. Dynamically importing `DualStackRNN` and mapping it to the `--model` choice `"dual_stack_rnn"`.
2. Restructuring `src/scripts/run_experiments.py` to add `copy` and `abc` task execution branches.
3. Creating training and evaluation loops on CPU using the generators from `src.data.context_sensitive`, utilizing `ignore_index = 3` for the ABC task to handle padding correctly.
4. Writing comparison statistics to `results_table.csv` using the columns `model,accuracy,token_accuracy,sequence_accuracy,sparsity` to ensure compatibility with downstream metrics.

The complete design proposal has been written to `/home/b/microgpt/.agents/explorer_stage15_3/design_proposal.md`.

---

## 5. Verification Method

To verify the proposed implementation once written:
1. Run the experimental script with mock configs for both new tasks to ensure execution completes successfully:
   - `python src/scripts/run_experiments.py --task copy --config mock --model all`
   - `python src/scripts/run_experiments.py --task abc --config mock --model all`
2. Check that the output directory contains `results_table.csv` and that the schema exactly matches:
   `model,accuracy,token_accuracy,sequence_accuracy,sparsity`
3. Run the project tests using `pytest tests/test_context_sensitive.py` to ensure task generators operate correctly.
