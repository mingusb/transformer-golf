# Design Proposal: Phase 3 Context-Sensitive Sequence Generators

This document outlines the precise specifications, signatures, mappings, tensor shapes, and mock examples for the two new sequence routing tasks required in Phase 3 (Stage 13):
1. **The Copy Task ($ww$)**
2. **Multiple Counting ($a^n b^n c^n$)**

Both generators will be implemented in `src/data/context_sensitive.py`.

---

## 1. The Copy Task ($ww$)

### Function Signature
```python
import torch
from typing import Tuple

def generate_copy_task(
    num_samples: int,
    length: int,
    vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences for the Copy Task (w # w) where w is a sequence of random tokens
    and '#' is a delimiter token.
    
    Parameters:
    -----------
    num_samples : int
        The number of sequence samples to generate. Must be > 0.
    length : int
        The length of the sub-sequence 'w'. The total sequence length returned is 2 * length + 1.
        Must be > 0.
    vocab_size : int
        The total size of the vocabulary (including the delimiter). Must be >= 2.
        
    Returns:
    --------
    inputs : torch.Tensor
        Sequence tensor of shape (num_samples, 2 * length + 1) of type torch.long.
    targets : torch.Tensor
        Left-shifted input sequences of shape (num_samples, 2 * length + 1) of type torch.long,
        with the last token set to 0.
    """
```

### Clarification of Parameters
- **`length`**: Represents the length of the sub-sequence $w$. The total length of the sequence is $2 \times \text{length} + 1$, which includes the delimiter token in the middle.
- **Delimiter Token**: If `vocab_size` is $V$, the tokens within the sub-sequence $w$ are drawn uniformly at random from the range $[0, V - 2]$. The delimiter token `#` is designated as the last token in the vocabulary: $V - 1$.
- **Constraint**: `vocab_size` must be at least 2. When `vocab_size = 2`, $w$ is comprised entirely of `0`s and the delimiter is `1`.

### Next-Token Prediction Target Formulation
The model performs next-token prediction, so the target sequence is a left-shifted version of the input sequence. 
- Input: `[w_0, w_1, ..., w_{L-1}, delimiter, w_0, w_1, ..., w_{L-1}]`
- Target: `[w_1, ..., w_{L-1}, delimiter, w_0, w_1, ..., w_{L-1}, 0]`
- The final token in the target sequence is set to `0` (matching the pattern in `generate_nested_brackets`).

### Tensor Shapes & Dtypes
- **`inputs`**: `shape = (num_samples, 2 * length + 1)`, `dtype = torch.long`
- **`targets`**: `shape = (num_samples, 2 * length + 1)`, `dtype = torch.long`

### Mock Example
- **Arguments**: `num_samples=1`, `length=3`, `vocab_size=4`
  - Vocabulary range: $0, 1, 2, 3$ (where $0, 1, 2$ represent content tokens, and $3$ is the delimiter).
  - Selected sub-sequence $w$: `[1, 0, 2]`
- **Inputs**: `[[1, 0, 2, 3, 1, 0, 2]]` (shape: `(1, 7)`)
- **Targets**: `[[0, 2, 3, 1, 0, 2, 0]]` (shape: `(1, 7)`)

---

## 2. Multiple Counting ($a^n b^n c^n$)

### Function Signature
```python
def generate_abc_task(
    num_samples: int,
    n_max: int,
    n: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences of the form a^n b^n c^n to test multiple counting.
    
    Parameters:
    -----------
    num_samples : int
        The number of sequence samples to generate. Must be > 0.
    n_max : int
        The maximum count of each character. Must be > 0.
    n : int, optional
        If specified, generates sequences with exactly 'n' of each character.
        If specified, total sequence length is 3 * n. Must satisfy 0 < n <= n_max.
        If None, 'n' is randomly sampled for each sequence from the range [1, n_max],
        and sequences are right-padded to 3 * n_max using the padding token.
        
    Returns:
    --------
    inputs : torch.Tensor
        Sequence tensor of shape (num_samples, 3 * n_max) [if n is None] 
        or (num_samples, 3 * n) [if n is specified] of type torch.long.
    targets : torch.Tensor
        Left-shifted input sequences of type torch.long. The last token is set to the padding token (3).
    """
```

### Clarification of Parameters & Mappings
- **Token Representation**:
  - `'a'` $\rightarrow$ `0`
  - `'b'` $\rightarrow$ `1`
  - `'c'` $\rightarrow$ `2`
  - `'PAD'` (padding / EOS token) $\rightarrow$ `3`
- **Varying $n$**:
  - When `n = None` (default training behavior), $n$ is randomly sampled for each sequence from the range $[1, n_{max}]$. To form a rectangular batch tensor, sequences of length $3n$ are right-padded with the `PAD` token (3) to a fixed length of $3 \times n_{max}$.
  - When `n` is specified (e.g. for length generalization evaluations), all sequences in the batch have length exactly $3n$, and no padding is applied to the input tensor.

### Next-Token Prediction Target Formulation
The targets are the left-shifted inputs.
- For padded sequences (`n = None`, sample $i$ with count $n_i$):
  - Input: `[0]*n_i + [1]*n_i + [2]*n_i + [3]*(3*n_max - 3*n_i)`
  - Target: `[0]*(n_i-1) + [1]*n_i + [2]*n_i + [3]*(3*n_max - 3*n_i) + [3]`
- For fixed-length sequences (specified `n`):
  - Input: `[0]*n + [1]*n + [2]*n`
  - Target: `[0]*(n-1) + [1]*n + [2]*n + [3]`

### Tensor Shapes & Dtypes
- **If `n = None`**:
  - `inputs`: `shape = (num_samples, 3 * n_max)`, `dtype = torch.long`
  - `targets`: `shape = (num_samples, 3 * n_max)`, `dtype = torch.long`
- **If `n` is specified**:
  - `inputs`: `shape = (num_samples, 3 * n)`, `dtype = torch.long`
  - `targets`: `shape = (num_samples, 3 * n)`, `dtype = torch.long`

### Mock Examples
#### Case A: `num_samples=1`, `n_max=3`, `n=None`
- Suppose $n_i = 2$ is randomly selected.
- **Inputs**: `[[0, 0, 1, 1, 2, 2, 3, 3, 3]]` (shape: `(1, 9)`)
- **Targets**: `[[0, 1, 1, 2, 2, 3, 3, 3, 3]]` (shape: `(1, 9)`)

#### Case B: `num_samples=1`, `n_max=3`, `n=2` (fixed)
- **Inputs**: `[[0, 0, 1, 1, 2, 2]]` (shape: `(1, 6)`)
- **Targets**: `[[0, 1, 1, 2, 2, 3]]` (shape: `(1, 6)`)

---

## 3. Testing Strategy

Unit tests will be added to `tests/test_context_sensitive.py` to ensure correctness of both generators:

1. **Shape & Dtype Verification**:
   - Verify outputs are PyTorch `long` tensors.
   - Verify shapes: `(num_samples, 2 * length + 1)` for copy task; `(num_samples, 3 * n_max)` or `(num_samples, 3 * n)` for $a^n b^n c^n$.

2. **Copy Task Correctness**:
   - Check that the sub-sequence before the delimiter matches the sub-sequence after.
   - Check that all content tokens are within range $[0, V - 2]$.
   - Check that the delimiter is exactly at the middle index `length` and has value $V - 1$.
   - Verify target sequence matches the left-shifted input with a trailing `0`.

3. **Multiple Counting Correctness**:
   - Check that the input sequence starts with a run of `0`s, followed by a run of `1`s, followed by a run of `2`s, all of equal length $n$.
   - Verify that any remaining positions are filled with `3` (the `PAD` token).
   - Verify that when `n` is specified, the sequence length is exactly $3n$ with no padding in the input.
   - Verify target sequence matches the left-shifted input with a trailing `3`.

4. **Invalid Argument Handling**:
   - Raise `ValueError` for invalid parameter values (e.g. non-positive values, `vocab_size < 2`, or `n > n_max`).
