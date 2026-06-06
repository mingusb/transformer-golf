# Original User Request

## Initial Request — 2026-06-06T01:21:59Z

<USER_REQUEST>
Implement Phase 3 of the "Transformer Golf" project: Context-Sensitive Routing & Universal Computation.

Working directory: `/home/b/microgpt`
Integrity mode: development

**Reference Material**: Read `/home/b/microgpt/phase_3_specs.md` before beginning. It defines the exact requirements for Stages 13 through 15.

## Requirements

### R1. Context-Sensitive Data Generation (Stage 13)
Implement `src/data/context_sensitive.py` to generate The Copy Task ($ww$) and Multiple Counting ($a^n b^n c^n$).
Formulate them as next-token prediction paths.

### R2. Universal Memory Architecture (Stage 14)
Implement a Turing-complete neural architecture in `src/models/universal_rnn.py`. You may choose either a Dual-Stack RNN or a Tape-Augmented RNN (Neural Turing Machine). It must be fully differentiable and able to learn context-sensitive logic.

### R3. Integration & Evaluation (Stage 15)
Update `src/scripts/run_experiments.py` to support `--task copy` and `--task abc`. 
Evaluate your Universal Architecture against the standard `StackRNN` from Phase 2. Ensure that the StackRNN mathematically fails (accuracy < 1.0) while the new Universal Architecture succeeds.

## Acceptance Criteria
- [ ] A test suite (`pytest`) runs successfully for all new components.
- [ ] The `run_experiments.py` script automatically evaluates all new conditions.
- [ ] The project passes internal E2E testing without any syntax or logic errors.
</USER_REQUEST>

## Follow-up — 2026-06-06T03:03:13Z

<USER_REQUEST>
Develop a theoretical architectural plan to solve the $a^n b^n c^n$ multiple-counting sequence problem using a Liquid State Machine (LSM). 

CRITICAL CONSTRAINT: You must strictly output the plan as a Markdown artifact (e.g., `lsm_universal_plan.md`). DO NOT WRITE ANY CODE. This is purely a theoretical architecture and planning phase.

Context: 
The $a^n b^n c^n$ task is a Context-Sensitive Language that requires universal computation (or at least two independent stacks). Standard LSMs fail due to fading memory and lack of discrete state tracking. 

Requirements for the Plan:
1. Incorporate state-of-the-art LSM principles (e.g., Temporal Excitation Partitioned Reservoir Ensembles (TEPRE), Multi-Scale Time Constants, Predictive Coding/RLSM, and Adaptive Spiking Gating).
2. Explicitly explain the mathematical or structural mechanism the LSM will use to overcome exponential fading memory.
3. Detail how the specific sequence states (tracking the count of 'a's, shifting to 'b's without destroying the 'a' count, and matching 'c's) will be routed through the spiking reservoirs.
4. Define the training protocol (e.g., localized STDP with eligibility traces or predictive coding) that will be used instead of Backpropagation Through Time (BPTT).
</USER_REQUEST>
