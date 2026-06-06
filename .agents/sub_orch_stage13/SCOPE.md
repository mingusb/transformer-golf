# Scope: Stage 13 - Context-Sensitive Sequence Generation

## Architecture
- Module: `src/data/context_sensitive.py`
- Functions to generate context-sensitive datasets.
- Tasks:
  1. The Copy Task ($ww$)
  2. Multiple Counting ($a^n b^n c^n$)

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Design generator interfaces | Define python function signatures for data generation | None | DONE |
| 2 | Implement copy task generator | Write generator in `src/data/context_sensitive.py` | Milestone 1 | DONE |
| 3 | Implement abc task generator | Write generator in `src/data/context_sensitive.py` | Milestone 2 | DONE |
| 4 | Verify generators | Run pytest tests to confirm correct shapes and logic | Milestone 3 | DONE |

## Interface Contracts
- `generate_copy_task(num_samples: int, length: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]`
- `generate_abc_task(num_samples: int, n_max: int) -> Tuple[torch.Tensor, torch.Tensor]`
- Inputs should be integer tensors representing token sequences.
- Targets should be left-shifted versions of the input sequences.
- Delimiter for copy task should be a distinct token (e.g., `vocab_size - 1` or specified).
- Vocab range for multiple counting: `a` -> 0, `b` -> 1, `c` -> 2, etc. (we need to be clear about the vocabulary mappings).

## Code Layout
- `src/data/context_sensitive.py`
