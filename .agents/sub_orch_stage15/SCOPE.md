# Scope: Stage 15 - Evaluation & Integration

## Architecture
- Module: `src/scripts/run_experiments.py`
- Enhance the command-line interface to support new tasks: `--task copy` and `--task abc`.
- Support `--model dual_stack_rnn` (and option to run all/compare).
- Train and evaluate `DualStackRNN` against `StackRNN` on Copy Task and ABC Task.
- Show that `DualStackRNN` achieves 1.0 (or near 1.0) sequence accuracy, while `StackRNN` fails.
- Save summary statistics to `results_table.csv` and generate plots / output reports as required.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Design CLI & Experiment logic | Propose script modifications (Conv: 918b3325-c130-42b6-9dfb-578e81389a08, 8017dced-107e-4665-9911-b80b2ee071ff, 14ba6ff6-d0ba-48a9-a451-33cab6a78778) | None | DONE |
| 2 | Modify run_experiments.py | Integrate tasks copy and abc, DualStackRNN model, and evaluation logic | Milestone 1 | DONE |
| 3 | Verify script execution | Run command line experiments and pytest tests to ensure everything passes (Conv: 5d957ef1-4a5f-48a6-98fa-7ca6f9bb024c, 35d1a8f7-3c13-4092-978e-25621d074759, 74193d74-ed4a-4a92-8fbf-e666bbdf7487, 4ae8c239-42aa-4b9f-a153-da4a712d8db6, 4f5d1399-05ef-4ee2-88d0-eea8ca351dcb) | Milestone 2 | DONE |

## Interface Contracts
- Command support:
  - `python src/scripts/run_experiments.py --task copy --config mock --model dual_stack_rnn`
  - `python src/scripts/run_experiments.py --task abc --config mock --model dual_stack_rnn`
- Output files:
  - Writes to `results_table.csv` in the output directory.
  - Gracefully handles CLI options and respects directories.

## Code Layout
- `src/scripts/run_experiments.py`
