# E2E Testing Track Plan

## Objectives
1. Design the E2E test infrastructure and document it in `/home/b/microgpt/TEST_INFRA.md`.
2. Implement a comprehensive test suite in `/home/b/microgpt/tests/test_phase_3.py` covering Tiers 1-4 for Phase 3 (Stages 13, 14, 15) with features:
   - F1: Copy Task Generator
   - F2: ABC Task Generator
   - F3: DualStackRNN
   - F4: run_experiments.py integration
3. Run tests using `pytest` to verify that they run (gracefully skipping features that are not yet implemented).
4. Publish `/home/b/microgpt/TEST_READY.md` once complete.

## Milestone decomposition
- **Milestone 1: Create TEST_INFRA.md**
  - Task: Document the E2E test design, architecture, features, and coverage thresholds.
  - Verification: `/home/b/microgpt/TEST_INFRA.md` exists and matches the spec.
- **Milestone 2: Implement Tiers 1-4 tests**
  - Task: Implement the pytest suite in `/home/b/microgpt/tests/test_phase_3.py`. Must contain at least 20 Tier 1 tests, 20 Tier 2 tests, 4 Tier 3 tests, and 5 Tier 4 tests.
  - Verification: `/home/b/microgpt/tests/test_phase_3.py` exists and is formatted correctly.
- **Milestone 3: Verify Test Suite & Publish TEST_READY.md**
  - Task: Run `pytest tests/test_phase_3.py` to ensure it passes/skips gracefully. Once verified, create `/home/b/microgpt/TEST_READY.md`.
  - Verification: Pytest exit code is 0 (all tests passing or skipped), and `TEST_READY.md` is successfully published.
