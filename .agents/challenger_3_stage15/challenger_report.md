# Stage 15 Verification Report

This report summarizes the empirical verification of the Stage 15 implementation in `src/scripts/run_experiments.py`, covering StackRNN vs DualStackRNN length generalization, output formatting of results, and invalid arguments handling.

---

## 1. StackRNN vs DualStackRNN Length Generalization

### 1.1 Theoretical Framework
* **Copy Task ($w \text{ \# } w$)**: A single LIFO (Last-In-First-Out) stack reverses the input sequence, outputting $w^R$. Copying a sequence in its original order requires a FIFO queue. A FIFO queue can be simulated using two LIFO stacks (by transferring elements from the first to the second stack, reversing them a second time). Therefore, theoretically, **StackRNN (single stack) cannot solve copy generalization**, whereas **DualStackRNN (two stacks) can**.
* **ABC Task ($a^n b^n c^n$)**: Recognising or generating $a^n b^n c^n$ requires counting the number of $a$s, matching it to the number of $b$s, and then matching it to the number of $c$s. A single stack can count $a$s and match them to $b$s (by popping), but it will then be empty and cannot match them to $c$s. A dual-stack architecture allows storing/copying the count to match both $b$s and $c$s. Thus, **StackRNN cannot solve ABC generalization**, while **DualStackRNN can**.

### 1.2 Empirical Results

#### Experiment A: Copy Task (Copied Portion Only)
We trained StackRNN and DualStackRNN on $N=150$ samples of length $\text{train\_len}=3$ (sequence length 7) for 400 epochs, and evaluated them under length generalization ($\text{test\_len} = 3, 4, 5, 6, 8$).
We measured sequence accuracy on the copied portion only (excluding the unpredictable prefix):
* **Length 3 (Train Length)**:
  * StackRNN Sequence Accuracy: **1.0000** (converged)
  * DualStackRNN Sequence Accuracy: **1.0000** (converged)
* **Length 4**:
  * StackRNN Sequence Accuracy: **0.2200**
  * DualStackRNN Sequence Accuracy: **0.2600**
* **Length 5**:
  * StackRNN Sequence Accuracy: **0.0200**
  * DualStackRNN Sequence Accuracy: **0.0800**
* **Length 6**:
  * StackRNN Sequence Accuracy: **0.0000**
  * DualStackRNN Sequence Accuracy: **0.0100**
* **Length 8**:
  * StackRNN Sequence Accuracy: **0.0000**
  * DualStackRNN Sequence Accuracy: **0.0000**

**Observation**: While both models converge on the training length, they both fail to generalize to longer sequences in practice. DualStackRNN exhibits slightly higher token and sequence accuracies at intermediate lengths (e.g. length 4), but ultimately drops to 0% sequence accuracy. This is due to the classic challenge of training soft stack architectures: gradient descent tends to fall into local minima where the controller GRU memorizes the fixed sequence length instead of learning stack manipulation policies.

#### Experiment B: ABC Task (Fixed vs. Variable $n$)
We evaluated the models on the ABC task:
1. **Fixed $n$** (as implemented in `run_experiments.py`):
   * Training set: `a a a a a b b b b b c c c c c` (only 1 unique sequence)
   * Test set: `a^15 b^15 c^15`
   * **Results**: Both models memorized the single training sequence (Train Seq Acc = 1.0000) but completely failed to generalize to $n=15$ (Test Seq Acc = 0.0000). Both achieved the exact same Test Token Acc of **0.5455** due to predicting a static prefix pattern.
2. **Variable $n$** (randomly sampled $n \in [1, 3]$):
   * **Results**: Neither model converged to 100% (Train Seq Acc = 0.3333, Test Seq Acc = 0.1200) within 250 epochs, highlighting the difficulty of learning discrete stack operations via continuous soft-attention gates.

---

## 2. Output Formatting in `results_table.csv`

We ran `run_experiments.py` for all tasks under `--config mock` to verify the output formatting:
* Header format matched **exactly**: `model,accuracy,token_accuracy,sequence_accuracy,sparsity`
* All numeric columns (`accuracy`, `token_accuracy`, `sequence_accuracy`, `sparsity`) were formatted to **exactly 4 decimal places** (e.g. `1.0000`, `0.2014`, `0.0000`).

Example generated CSV content:
```csv
model,accuracy,token_accuracy,sequence_accuracy,sparsity
SSM,1.0000,1.0000,1.0000,0.2014
Attention,0.5000,0.5000,0.0000,0.0000
Conv1D,1.0000,1.0000,1.0000,0.0000
MarkovChain,1.0000,1.0000,1.0000,0.0000
LSM,0.9600,0.9600,0.6000,0.0000
```

---

## 3. Invalid Arguments Handling

We tested `run_experiments.py` with invalid arguments:
1. **Invalid Task**: `python3 src/scripts/run_experiments.py --task invalid_task`
   * Output: `run_experiments.py: error: argument --task: invalid choice: 'invalid_task' (choose from alternating, nesting, copy, abc)`
   * Exit Code: `2` (graceful rejection by argparse)
2. **Invalid Model**: `python3 src/scripts/run_experiments.py --model invalid_model`
   * Output: `run_experiments.py: error: argument --model: invalid choice: 'invalid_model' (choose from all, ssm, attention, conv1d, markov, stack_rnn, lsm, dual_stack_rnn)`
   * Exit Code: `2` (graceful rejection by argparse)
3. **Unrecognized Option**: `python3 src/scripts/run_experiments.py --foo bar`
   * Output: `run_experiments.py: error: unrecognized arguments: --foo bar`
   * Exit Code: `2` (graceful rejection by argparse)
