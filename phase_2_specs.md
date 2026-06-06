# Transformer Golf: Phase 2 Specs
## The Hierarchical Sequence Nesting Benchmark

Following the successful completion of the 8-stage pipeline for spatial and temporal sequence routing (DFA tracking), Phase 2 will focus on the theoretical limitations of both Recurrent SSMs and Causal Attention models when presented with Unbounded Hierarchical Nesting.

### Stage 9: Hierarchical Sequence Nesting Data Generation
**Objective**: Generate a dataset of perfectly balanced, deeply nested sequences using $n$ different types of brackets (e.g., `(`, `)`, `[`, `]`, `{`, `}`).
**Implementation Details**:
- Create `src/data/nested_brackets.py`.
- The generator must produce sequences where the next valid closing bracket is strictly determined by the most recently opened, unclosed bracket (a LIFO stack structure).
- Sequences will be generated at varying depths (e.g., depth 1 to depth 50) to evaluate the capacity thresholds of the models.

### Stage 10: Depth-vs-Accuracy Profiling
**Objective**: Empirically demonstrate that fixed-size continuous states (SSM) and fixed-depth attention layers (Transformers) fundamentally fail at Unbounded Hierarchical Nesting.
**Implementation Details**:
- Update `src/scripts/run_experiments.py` to support `--task nesting`.
- Evaluate the previously trained `RecurrentSSM` and `CausalAttention` models on the new dataset.
- Generate a new metric plot/table: **Accuracy vs. Nesting Depth**. We expect both models to succeed at shallow depths and catastrophically fail at deep nesting due to capacity overflow and lack of sequential reasoning depth.

### Stage 11: The Stack-Augmented Control Baseline
**Objective**: Introduce an architecture mathematically capable of solving the task to serve as the absolute baseline control, continuing the "Transformer Golf" methodology.
**Implementation Details**:
- Implement a `StackRNN` in `src/models/stack_rnn.py`.
- This model will be augmented with a differentiable Pushdown Stack (a memory tape that can push and pop discrete states).
- Train the `StackRNN` and prove that it achieves 100% accuracy across all nesting depths (infinite depth capacity), successfully "golfing" the architecture to match the formal language complexity of the task.
