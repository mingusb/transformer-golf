# Phase 3: Context-Sensitive Routing & Universal Computation

## Overview
Phase 1 explored continuous state spaces (SSM) against finite state tracking (DFA). Phase 2 explored pushdown memory (Stack-RNN) against context-free languages (Dyck-$n$). 
Phase 3 evaluates sequence routing for tasks that require **Universal Computation (Turing Completeness)**. Specifically, tasks that exceed the capacity of a Pushdown Automaton (PDA).

## Stage 13: Context-Sensitive Sequence Generation
We define two new sequence routing tasks that cannot be solved by a single LIFO stack:
1. **The Copy Task ($ww$)**: The network receives a sequence of tokens followed by a delimiter, and must output the exact same sequence in the same order (FIFO logic).
2. **Multiple Counting ($a^n b^n c^n$)**: The network must track a count $n$, output $n$ `a`s, then $n$ `b`s, and then $n$ `c`s. A single stack is emptied after matching the `b`s and cannot match the `c`s.

**Requirements**:
- Implement generators for these two tasks in `src/data/context_sensitive.py`.
- Formulate them as next-token prediction tasks (similar to Phase 2).

## Stage 14: Universal Memory Architectures
To solve these tasks, implement a sequence model capable of Universal Computation.

**Option A: Dual-Stack RNN**
A Pushdown Automaton with two independent stacks is Turing-complete. Implement `DualStackRNN` by extending the `StackRNN` logic to project two separate push/pop operations for two independent differentiable continuous stacks.

**Option B: Tape-Augmented RNN (Neural Turing Machine)**
Implement an RNN with a differentiable memory matrix (tape) and location-based or content-based addressing read/write heads.

**Requirements**:
- Implement the model in `src/models/universal_rnn.py`.
- The model must be able to achieve 1.0000 sequence accuracy on both the Copy Task and the $a^n b^n c^n$ task when trained to convergence.

## Stage 15: Evaluation & Integration
- Update `src/scripts/run_experiments.py` to support `--task copy` and `--task abc`.
- Compare the new Universal model against the standard `StackRNN` from Phase 2.
- The `StackRNN` should mathematically fail (accuracy bound < 1.0), while the Universal model succeeds.

## Deliverables
- `src/data/context_sensitive.py`
- `src/models/universal_rnn.py`
- Integration into `src/scripts/run_experiments.py`
- Passing `pytest` tests.
