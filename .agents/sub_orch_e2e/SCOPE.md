# Scope: E2E Testing Track

## Architecture
- Define opaque-box testing framework for Phase 3 of Transformer Golf.
- The test suite must run using `pytest` (e.g., written in `tests/test_phase_3.py`).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Create TEST_INFRA.md | Document test design, features, and layout | None | DONE |
| 2 | Implement Tiers 1-4 tests | Write test cases following 4-tier methodology in `tests/test_phase_3.py` | None | DONE |
| 3 | Verify Test Suite | Run tests and publish `TEST_READY.md` | Milestone 2 | DONE |

## Interface Contracts
- Opaque-box tests must verify `src/data/context_sensitive.py`, `src/models/universal_rnn.py`, and `src/scripts/run_experiments.py` from the outside.
- Test runner must execute tests via standard `pytest`.

## Code Layout
- `tests/test_phase_3.py` - Test suite containing all Tier 1-4 test cases.
