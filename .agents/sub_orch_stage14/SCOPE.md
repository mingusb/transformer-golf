# Scope: Stage 14 - Universal Memory Architectures

## Architecture
- Module: `src/models/universal_rnn.py`
- Model: `DualStackRNN` which is a PyTorch neural network that implements a Turing-complete RNN with two independent differentiable continuous stacks.
- We will extend `StackRNN` logic to two independent stacks.
- The model must have attributes `stack_width` and `stack_depth` and accept `(vocab_size, hidden_size, stack_width, stack_depth)` in its constructor.
- Its `forward` pass must take `(x, state=None)` and return `(logits, (stack1_states, stack2_states))`.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Design Dual-Stack RNN | Propose architecture design and operation gating logic in `.agents/explorer_stage14/design_proposal.md` | None | DONE (c596233f-8832-47bc-a053-15bf734dfa53) |
| 2 | Implement DualStackRNN | Write the code in `src/models/universal_rnn.py` | Milestone 1 | DONE (1d6b9571-1f8e-4e37-82f2-58635ea5ed25) |
| 3 | Verify DualStackRNN | Run unit tests and confirm shape, gradients flow, differentiability | Milestone 2 | DONE (18cf2605-10a5-4f41-9569-e4baf5f0ef34, 7af0d799-c5bf-4174-9502-75c183fe127b, 6a46e459-8183-44ae-b772-6ef06871e43b, 76e613d7-6e1b-467d-88ee-60bbbee06187, 8dc94a19-edff-4937-a73c-377c06c7c67e) |

## Interface Contracts
- Constructor: `DualStackRNN(vocab_size: int, hidden_size: int, stack_width: int, stack_depth: int)`
- Forward: `forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
- State return: `(logits, (stack1_states, stack2_states))` where each stack state tensor has shape `(batch_size, seq_len, stack_depth, stack_width)`.

## Code Layout
- `src/models/universal_rnn.py`
