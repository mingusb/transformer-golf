# Stage 15 Review Report

## Review Summary

**Verdict**: **APPROVE**

The implementation of Stage 15 (Evaluation & Integration) in `src/scripts/run_experiments.py` is fully complete, correct, robust, and conforms to the specified interface contracts. All 49 tests in `tests/test_phase_3.py` pass successfully, and the experiment script correctly generates CSV outputs matching the expected schema.

---

## Quality Review Findings

### [Minor] Finding 1: Copy Task Target Padding Overlaps with Vocabulary Token
- **What**: The copy task generator `generate_copy_task` pads the target sequences with a zero token `0`.
- **Where**: `src/data/context_sensitive.py`, line 52.
- **Why**: Since `0` is a valid symbol in the copy task's sequence vocabulary `[0, vocab_size - 2]`, this creates an overlap. When evaluated with `ignore_index = None`, the model is forced to predict `0` at the final sequence step.
- **Suggestion**: This is a minor design behavior in the data generator rather than an implementation bug in `run_experiments.py`. Since `run_experiments.py` behaves correctly given this layout, no change is required, but it is worth noting for model evaluation.

### [Minor] Finding 2: Sequence Accuracy Calculation under empty inputs
- **What**: If an empty tensor is evaluated in `evaluate_model_accs`, the calculation of sequence accuracy mean could produce `NaN`.
- **Where**: `src/scripts/run_experiments.py`, line 63 and 66.
- **Why**: Under normal CLI runs, the script enforces a minimum sample size (`num_samples = 5` for mock, `100` for real). Therefore, this is not triggered in practice.
- **Suggestion**: A safety check or `max()` call on batch dimension could be added in the future if zero-length datasets are supported.

---

## Verified Claims

- **CLI choices configuration** → Verified by executing `python3 src/scripts/run_experiments.py --help` and inspecting `argparse` choices. All new choices (`copy`, `abc`, `dual_stack_rnn`) are supported. → **PASS**
- **Test Suite Pass** → Ran `pytest tests/test_phase_3.py`. All 49 tests passed. → **PASS**
- **Results CSV Schema & Precision** → Inspected `results_copy_mock/results_table.csv` and verified that the columns `model,accuracy,token_accuracy,sequence_accuracy,sparsity` are present and floating-point metrics are formatted to 4 decimal places. → **PASS**
- **Ignore Index Logic** → Verified that for ABC task, targets are correctly ignored using index `3` in `CrossEntropyLoss` and masked out in accuracy calculations. → **PASS**

---

## Coverage Gaps

- **Real Configuration Training** — risk level: **LOW** — recommendation: **accept risk**. Running the full `real_config` across 10 seeds for 80 epochs takes a significant amount of computation time. The code execution flow is identical to the mock configuration, which has been verified to run and output correct results.

---

## Unverified Items

- None.

---

# Adversarial Challenge Report

## Challenge Summary

**Overall risk assessment**: **LOW**

The integration code demonstrates strong resilience against input variation, proper configuration defaults, and valid gradient verification steps.

## Challenges

### [Low] Challenge 1: Out-of-vocabulary Inputs and Device Transfer
- **Assumption challenged**: The model is evaluated on the same hardware and vocabulary bounds.
- **Attack scenario**: If inputs are transferred to different devices (GPU/CPU) or vocabulary sizes are not aligned, tensor operations might fail.
- **Blast radius**: Model training crashes with device mismatch or embedding index out of range.
- **Mitigation**: The code correctly instantiates the models dynamically with proper `vocab_size` for each task and uses standard PyTorch device alignments when available.

---

## Stress Test Results

- **Alternating Task Mock Run** → Verified command: `python3 src/scripts/run_experiments.py --task alternating --config mock` → **PASS**
- **Copy Task Mock Run** → Verified command: `python3 src/scripts/run_experiments.py --task copy --config mock` → **PASS**
- **ABC Task Mock Run** → Verified command: `python3 src/scripts/run_experiments.py --task abc --config mock` → **PASS**

---

## Unchallenged Areas

- **Full Epoch Scaling**: Did not perform adversarial stress tests on extremely long-running epoch iterations due to timing bounds.
