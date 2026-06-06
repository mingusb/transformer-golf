# E2E Test Infra: Transformer Golf Phase 3

## Test Philosophy
- Opaque-box, requirement-driven. No dependency on implementation design.
- Methodology: Category-Partition + BVA + Pairwise + Workload Testing.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | F1: Copy Task Generator | Phase 3 Specs §Stage 13 | 5      | 5      | ✓      |
| 2 | F2: ABC Task Generator  | Phase 3 Specs §Stage 13 | 5      | 5      | ✓      |
| 3 | F3: DualStackRNN        | Phase 3 Specs §Stage 14 | 5      | 5      | ✓      |
| 4 | F4: run_experiments.py  | Phase 3 Specs §Stage 15 | 5      | 5      | ✓      |

## Test Architecture
- Test runner: `pytest`
- Test case format: pytest unit tests with custom parameters, using dynamic imports to skip gracefully when implementation is not yet completed.
- Directory layout: `tests/test_phase_3.py`

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Copy sequence processing | F1, F3, F4 | Medium |
| 2 | ABC counting sequence processing | F2, F3, F4 | Medium |
| 3 | Comparison: StackRNN vs DualStackRNN on Copy Task | F1, F3, F4 | High |
| 4 | Comparison: StackRNN vs DualStackRNN on ABC Task | F2, F3, F4 | High |
| 5 | General CLI multi-task validation | F1, F2, F3, F4 | High |

## Coverage Thresholds
- Tier 1: ≥5 per feature
- Tier 2: ≥5 per feature (where boundaries exist)
- Tier 3: pairwise coverage of major feature interactions
- Tier 4: ≥5 realistic application scenarios
