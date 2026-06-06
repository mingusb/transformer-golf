# Project: Universal Liquid State Computing

## Architecture
- **Input Encoding**: Spikes are generated from discrete tokens (a, b, c, padding).
- **Adaptive Spiking Gating & Routing Layer**: Routes spike trains dynamically between transient sub-reservoirs and the persistent Integrate-and-Fire reservoir based on the current grammatical phase.
- **Multi-Scale Sub-Reservoirs**: Partitions neurons into transient sub-reservoirs with logarithmically scaled membrane time constants, and a persistent memory sub-reservoir using IF neurons.
- **Readout Layer**: Decodes the reservoir state to vocabulary logits.
- **Training Protocol (RLSM)**: Weights of the gating and readout layers are updated using localized D-STDP with eligibility traces and predictive coding feedback. Autograd is strictly disabled.

## Code Layout
- `src/models/universal_lsm.py`: Contains the `UniversalLSM` module, gating layer, sub-reservoir partitions, and custom training logic.
- `src/scripts/run_experiments.py`: Entry point for evaluating the models on the `abc` and other tasks.
- `tests/test_phase_4.py`: Unit and integration test suite verifying LSM dynamics, gating, no-BPTT gradients, and convergence.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Test Suite & Architecture Design | Create unit tests verifying constraints (no-BPTT, persistence, routing) | None | DONE |
| 2 | Multi-Scale Sub-Reservoirs & Adaptive Gating | Implement `UniversalLSM` architecture in `src/models/universal_lsm.py` | M1 | DONE |
| 3 | Pure Spike-driven RLSM Training | Implement D-STDP, eligibility traces, and predictive coding feedback | M2 | DONE |
| 4 | CLI Integration & abc Task Evaluation | Update `run_experiments.py` to evaluate UniversalLSM and match 1.0000 bound | M3 | IN_PROGRESS |
| 5 | E2E Verification & Forensic Audit | Run full test suites and perform forensics validation | M4 | PLANNED |

## Interface Contracts
### UniversalLSM API
- **Class**: `UniversalLSM(input_size: int, reservoir_size: int, output_size: int, ...)`
- **Forward Signature**: `forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]`
  - `x`: Long tensor of shape `(batch_size, seq_len)`
  - Returns `(logits, state)` where `logits` has shape `(batch_size, seq_len, output_size)`.
- **Training Method**: `fit(self, X: torch.Tensor, Y: torch.Tensor, epochs: int, lr: float, ignore_index: int)`
  - Updates weights of gating layer and readout matrix using spike-driven gradient-free learning.
