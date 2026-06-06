# Handoff Report - Stage 13 Data Generator Design

## 1. Observation
- **Phase 3 Specs (`phase_3_specs.md`, lines 7-15)**:
  ```markdown
  ## Stage 13: Context-Sensitive Sequence Generation
  We define two new sequence routing tasks that cannot be solved by a single LIFO stack:
  1. **The Copy Task ($ww$)**: The network receives a sequence of tokens followed by a delimiter, and must output the exact same sequence in the same order (FIFO logic).
  2. **Multiple Counting ($a^n b^n c^n$)**: The network must track a count $n$, output $n$ `a`s, then $n$ `b`s, and then $n$ `c`s. A single stack is emptied after matching the `b`s and cannot match the `c`s.

  **Requirements**:
  - Implement generators for these two tasks in `src/data/context_sensitive.py`.
  - Formulate them as next-token prediction tasks (similar to Phase 2).
  ```
- **Nesting Brackets Generator (`src/data/nested_brackets.py`, lines 95-97)**:
  ```python
          # Target is left-shifted input, with the last token set to 0 (first token of vocab)
          target = seq[1:] + [0]
  ```
- **Nesting Brackets Unit Tests (`tests/test_nested_brackets.py`, lines 7-14)**:
  ```python
  def test_generate_nested_brackets_shapes():
      """Test output shapes and types."""
      inputs, targets = generate_nested_brackets(num_samples=15, length=30, depth=5, num_bracket_types=3)
      assert isinstance(inputs, torch.Tensor)
      assert isinstance(targets, torch.Tensor)
      assert inputs.shape == (15, 30)
      assert targets.shape == (15, 30)
      assert inputs.dtype == torch.long
      assert targets.dtype == torch.long
  ```
- **DFA Stress Tests (`tests/test_dfa_stress.py`)**: Succeeded with 2 passed tests under pytest.
- **Nesting Brackets Tests (`tests/test_nested_brackets.py`)**: Succeeded with 4 passed tests under pytest.

## 2. Logic Chain
- Next-token prediction requires targets to be left-shifted inputs (based on the pattern in `src/data/nested_brackets.py`).
- For the Copy Task ($ww$):
  - Total sequence length must be $2 \times L + 1$, where $L$ is the length of $w$ (parameter `length`), to include the delimiter in the middle.
  - With vocabulary size $V$, the tokens within $w$ must be randomly drawn from $[0, V-2]$, leaving $V-1$ as the delimiter token.
  - The target is `inputs[:, 1:]` left-shifted, with the final element padded with `0`.
- For the Multiple Counting Task ($a^n b^n c^n$):
  - Tokens are mapped: `'a' -> 0`, `'b' -> 1`, `'c' -> 2`, `'PAD' -> 3`.
  - When varying $n$, we randomly select $n \in [1, n_{max}]$ for each sample, and right-pad the sequences to length $3 \times n_{max}$ with the `PAD` token (3).
  - To support length generalization training and testing, an optional parameter `n` is introduced. When specified, sequences are of exact length $3n$ without padding.
  - The target is `inputs[:, 1:]` left-shifted, with the final element padded with `3` (the `PAD` token).

## 3. Caveats
- Assumed that `vocab_size` for copy task is at least 2.
- Assumed that for Multiple Counting, standard representation of alphabet tokens (0, 1, 2) and padding (3) is acceptable and does not conflict with other parts of the project.
- Did not run/test integration with `DualStackRNN` since implementation is deferred to Stage 14.

## 4. Conclusion
- A precise design proposal has been formulated and written to `/home/b/microgpt/.agents/explorer_stage13/design_proposal.md`.
- Both functions `generate_copy_task` and `generate_abc_task` have complete signatures, token mappings, delimiter choices, tensor shapes, and mock examples defined.

## 5. Verification Method
- Code correctness of the proposed signatures and behaviors can be verified by checking `design_proposal.md`.
- Once implemented by the implementer agent, the generators should pass the unit tests described in the "Testing Strategy" section of `design_proposal.md` using `pytest tests/test_context_sensitive.py`.
- Run `pytest tests/test_nested_brackets.py` to ensure existing tests in the workspace remain functional.
