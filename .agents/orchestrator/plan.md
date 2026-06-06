# Project: Transformer Golf - Phase 3 Plan

## Architecture
- **Stage 13**: Context-sensitive sequence generation.
  - File: `src/data/context_sensitive.py`
  - Logic: Generates Copy Task ($ww$) and Multiple Counting ($a^n b^n c^n$). Formulated as next-token prediction tasks.
- **Stage 14**: Universal Memory Architecture.
  - File: `src/models/universal_rnn.py`
  - Logic: Implement a Turing-complete model. We will implement `DualStackRNN`, extending the `StackRNN` logic to two independent, differentiable, continuous stacks.
- **Stage 15**: Integration and Evaluation.
  - File: `src/scripts/run_experiments.py`
  - Logic: Train both models and print results. Save comparison tables/plots.

## Milestones
| # | Name | Scope | Dependencies | Status | Conv ID |
|---|------|-------|-------------|--------|---------|
| 1 | E2E Testing Track | Design and implement opaque-box test suite covering Tiers 1-4. Publish `TEST_READY.md`. | None | DONE | bd7fa6d4-6eb1-4ece-9d0c-31484aa4a026 |
| 2 | Stage 13 Implementation | Implement context-sensitive generators in `src/data/context_sensitive.py` | None | DONE | 65e76e28-ae15-4775-bd8a-0ad8ac6ce080 |
| 3 | Stage 14 Implementation | Implement differentiable `DualStackRNN` in `src/models/universal_rnn.py` | Stage 13 | DONE | f6370582-87f0-4269-9ce9-49715f796a4c |
| 4 | Stage 15 Integration | Update `src/scripts/run_experiments.py` and run comparisons | Stage 13, 14 | IN_PROGRESS | be07affa-8191-482a-93d6-5623b4aa2e0a |
| 5 | Phase 2 Adversarial Hardening | Tier 5 white-box testing and coverage validation | Stage 15 | PLANNED | TBD |

## Interface Contracts
- `src/data/context_sensitive.py`:
  - `generate_copy_task(num_samples, length, vocab_size)`: returns `(inputs, targets)` where targets are left-shifted.
  - `generate_abc_task(num_samples, n_max)`: returns `(inputs, targets)` where targets are left-shifted.
- `src/models/universal_rnn.py`:
  - `DualStackRNN(vocab_size, hidden_size, stack_width, stack_depth)`: extends `nn.Module`. Forward returns `(logits, stack_states)` where `stack_states` tracks both stacks.

## Code Layout
- `src/data/context_sensitive.py` - Context-sensitive data generation
- `src/models/universal_rnn.py` - Differentiable Turing-complete RNN model
- `src/scripts/run_experiments.py` - Entrypoint for running the evaluations
- `tests/test_phase_3.py` - Opaque-box test suite for Phase 3
