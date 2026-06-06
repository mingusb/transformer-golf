# Handoff Report: DualStackRNN Design Investigation

This handoff report summarizes the findings, reasoning, and design proposal for the `DualStackRNN` architecture.

## 1. Observation

- **`src/models/stack_rnn.py`**:
  - Defines `StackRNN` at line 5:
    ```python
    class StackRNN(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int, stack_width: int, stack_depth: int):
    ```
  - Initializes RNN cell taking concatenated input and stack top at line 19:
    ```python
    self.rnn_cell = nn.GRUCell(hidden_size + stack_width, hidden_size)
    ```
  - Soft stack updates computed using `stack_proj` at lines 73-90:
    ```python
    proj = self.stack_proj(h)  # shape: (batch_size, 3 + stack_width)
    ...
    S = p_t * S_push + o_t * S_pop + (1.0 - p_t - o_t) * S_prev
    ```

- **`tests/test_phase_3.py`**:
  - Instantiates `DualStackRNN` with signature `(vocab_size, hidden_size, stack_width, stack_depth)` at line 202:
    ```python
    model = DualStackRNN(vocab_size, hidden_size, stack_width, stack_depth)
    ```
  - Verifies class/instance attributes `stack_width` and `stack_depth` at lines 256-257:
    ```python
    assert model.stack_width == stack_width
    assert model.stack_depth == stack_depth
    ```
  - Verifies forward pass returns logits and a tuple/list of two tensors of shape `(batch_size, seq_len, stack_depth, stack_width)` at lines 204-208:
    ```python
    logits, stack_states = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    if isinstance(stack_states, (list, tuple)):
        for s in stack_states:
            assert s.shape == (batch_size, seq_len, stack_depth, stack_width)
    ```
  - Verifies that zero/negative dimensions raise `ValueError` at lines 390-394.

---

## 2. Logic Chain

1. From `tests/test_phase_3.py` (lines 202, 256-257), the `DualStackRNN` constructor must accept `(vocab_size, hidden_size, stack_width, stack_depth)` and expose attributes `stack_width` and `stack_depth` on the model object.
2. From `tests/test_phase_3.py` (lines 204-208), the forward pass of `DualStackRNN` must accept `(x, state=None)` and return `(logits, (stack1_states, stack2_states))`, where `logits` has shape `(batch_size, seq_len, vocab_size)` and `stack1_states` and `stack2_states` are tensors of shape `(batch_size, seq_len, stack_depth, stack_width)`.
3. To support two independent stacks, the controller GRU cell input size must be `hidden_size + 2 * stack_width` (incorporating both stack tops).
4. Each stack must be updated independently via separate linear projection layers (`self.stack1_proj` and `self.stack2_proj`) of output dimension `3 + stack_width` applied to the hidden state $h$.
5. The proposed class structure and mathematical updates in `/home/b/microgpt/.agents/sub_orch_stage14/design_proposal.md` successfully satisfy all of these constraints and requirements.

---

## 3. Caveats

- No caveats.

---

## 4. Conclusion

The architectural design for `DualStackRNN` has been completed. It extends the single-stack `StackRNN` by utilizing two independent differentiable stacks updated in parallel, feeding both top vectors back into the GRU controller cell. The complete implementation code has been drafted and saved in `/home/b/microgpt/.agents/sub_orch_stage14/design_proposal.md`.

---

## 5. Verification Method

To independently verify the implementation when it is written to `src/models/universal_rnn.py`:
1. Check the import at the top of `tests/test_phase_3.py`: `from src.models.universal_rnn import DualStackRNN`.
2. Run unit tests using:
   ```bash
   pytest tests/test_phase_3.py -k "DualStackRNN"
   ```
3. Ensure that all 10 tests for `DualStackRNN` pass without errors or warnings.
