# BRIEFING — 2026-06-06T01:23:18Z

## Mission
Formulate a precise design specification for the Copy Task (ww) and Multiple Counting (a^n b^n c^n) tasks, document existing testing patterns, and produce a design proposal.

## 🔒 My Identity
- Archetype: Teamwork explorer
- Roles: Read-only investigator, analyzer
- Working directory: /home/b/microgpt/.agents/explorer_stage13
- Original parent: 65e76e28-ae15-4775-bd8a-0ad8ac6ce080
- Milestone: Stage 13 Data Generator Design

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Code-only network mode
- Write only to our own directory `/home/b/microgpt/.agents/explorer_stage13`

## Current Parent
- Conversation ID: 65e76e28-ae15-4775-bd8a-0ad8ac6ce080
- Updated: 2026-06-06T01:24:00Z

## Investigation State
- **Explored paths**:
  - `phase_3_specs.md` (specifications for phase 3 sequence routing tasks)
  - `src/data/nested_brackets.py` (nesting generator code patterns)
  - `src/data/dfa.py` (DFA sequence generator and test-labels code patterns)
  - `tests/test_nested_brackets.py` (nesting generator unit tests)
  - `tests/test_dfa_stress.py` (DFA generator stress/validation tests)
  - `src/scripts/run_experiments.py` (integration context for generators and evaluations)
  - Phase 2 tests (`test_phase_2.py`, `test_phase_2_challenger_1.py`, `test_phase_2_challenger_2.py`)
- **Key findings**:
  - Existing next-token prediction targets are formatted by left-shifting the inputs and padding the last token with 0 or a dummy value.
  - The Copy Task is best defined with total sequence length $2L+1$ where $L$ is `length`, content vocab $[0, V-2]$, and delimiter $V-1$.
  - The Multiple Counting Task needs to vary $n \in [1, n_{max}]$, which requires padding using an EOS/padding token (token `3`). To support length generalization training and evaluation, an optional `n` parameter was designed to allow generation of fixed-length sequences (no padding, length $3n$).
  - All 76 Phase 2 tests pass successfully, confirming model and training pipelines are healthy.
- **Unexplored areas**: None.

## Key Decisions Made
- Formulate `generate_copy_task` using $2L+1$ total sequence length, delimiter token $V-1$, and content range $[0, V-2]$.
- Formulate `generate_abc_task` with tokens `'a' -> 0`, `'b' -> 1`, `'c' -> 2`, and `'PAD' -> 3`.
- Added optional `n` parameter to `generate_abc_task` to allow fixed-length sequences of length $3n$ for testing length generalization.

## Artifact Index
- `/home/b/microgpt/.agents/explorer_stage13/design_proposal.md` — Precise function signatures, token mappings, tensor shapes, mock inputs/targets, and testing strategies.
