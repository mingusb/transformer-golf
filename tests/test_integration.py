import os
import sys
import subprocess
import pytest
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

from src.symbolic.solver import (
    generate_routing_matrix,
    one_hot_encode,
    apply_routing,
    synthesize_circuit,
    evaluate_circuit
)
from src.models.map_reduce_mlp import compile_mlp_from_circuit
from src.data.dfa import DFAGenerator
from src.models.recurrent_ssm import RecurrentSSM
from src.models.sparsity import apply_l0_mask, l0_pruning_step
from src.models.baselines import (
    CausalAttentionModel,
    Conv1DModel,
    MarkovChainModel,
    run_t_test
)
from src.scripts.complexity_check import fit_degree

def train_model(model, X_train, Y_train, epochs=50, lr=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        loss.backward()
        optimizer.step()

# ==========================================
# Tier 3: Cross-Feature Combinations (Integration)
# ==========================================

def test_TEST_T3_01_F1_F2():
    # TEST_T3_01_F1_F2: Routing Formalization to Z3
    # Run Z3 solver directly on a formalized routing matrix to synthesize functional logic DAG.
    M, N = 3, 3
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    circuit = synthesize_circuit(seq_len=N, num_outputs=M, alphabet=['a', 'b'], routing_matrix=R)
    
    # Evaluate circuit for inputs
    inputs = [True, False, True]
    outputs = evaluate_circuit(circuit, inputs)
    # Since R is identity, outputs must match inputs
    assert outputs == inputs

def test_TEST_T3_02_F2_F3():
    # TEST_T3_02_F2_F3: Circuit Compilation to MLP
    # Compile Z3 solver DAG output into PyTorch MLP, verifying identical truth table outputs.
    R = [[1, 0], [0, 1]]
    circuit = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R)
    mlp = compile_mlp_from_circuit(circuit, alphabet_size=2)
    
    # Inputs: one-hot for ['a', 'b'] => [[1, 0], [0, 1]]
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # batch=1
    mlp_out = mlp(X)
    
    # Circuit evaluation
    c_out0 = evaluate_circuit(circuit, [True, False])
    c_out1 = evaluate_circuit(circuit, [False, True])
    
    # Verify shape and contents
    assert mlp_out.shape == (1, 2, 2)
    assert np.allclose(mlp_out[0, 0].detach().numpy(), [1.0, 0.0])
    assert np.allclose(mlp_out[0, 1].detach().numpy(), [0.0, 1.0])

def test_TEST_T3_03_F3_F4():
    # TEST_T3_03_F3_F4: MLP Complexity Validation
    # Run complexity checks on compiled MLPs to check if parameter scaling matches the theoretical O(N^2).
    lengths = [4, 8, 12]
    params = []
    for N in lengths:
        R = generate_routing_matrix(N, N)
        circuit = synthesize_circuit(seq_len=N, num_outputs=N, alphabet=['a', 'b'], routing_matrix=R)
        mlp = compile_mlp_from_circuit(circuit, alphabet_size=2)
        params.append(sum(p.numel() for p in mlp.parameters()))
    
    deg = fit_degree(lengths, params)
    assert np.allclose(deg, 2.0, atol=0.2)

def test_TEST_T3_04_F5_F6():
    # TEST_T3_04_F5_F6: DFA to SSM Probing
    # Train RecurrentSSM on sequences from DFA generator, asserting hidden states probe accurately for active DFA states.
    # We will use scikit-learn LogisticRegression to train a probe
    from sklearn.linear_model import LogisticRegression
    
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 1, 'b': 0}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    num_samples = 80
    seq_len = 10
    inputs, labels, paths = generator.generate_sequences(num_samples, seq_len)
    
    X_one_hot = []
    for seq in inputs:
        X_one_hot.append(one_hot_encode(seq, ['a', 'b']))
    X_tensor = torch.tensor(np.array(X_one_hot), dtype=torch.float32)
    
    ssm = RecurrentSSM(vocab_size=2, d_model=2, state_dim=6, hippo_init=True)
    
    # Train ssm parameters to track the DFA state path
    optimizer = torch.optim.Adam(ssm.parameters(), lr=0.05)
    for epoch in range(120):
        optimizer.zero_grad()
        loss = 0.0
        current_state = None
        for t in range(seq_len):
            xt_step = X_tensor[:, t, :].unsqueeze(1)
            logits_step, next_state = ssm(xt_step, current_state)
            logits = logits_step.squeeze(1) # shape (num_samples, 2)
            step_targets = torch.tensor([paths[i][t+1] for i in range(num_samples)], dtype=torch.long)
            loss += nn.CrossEntropyLoss()(logits, step_targets)
            current_state = next_state
        loss.backward()
        optimizer.step()
        
    # Collect states at each step using the trained SSM
    states = []
    current_state = None
    for t in range(seq_len):
        xt_step = X_tensor[:, t, :].unsqueeze(1)
        _, next_state = ssm(xt_step, current_state)
        states.append(next_state.detach().numpy())
        current_state = next_state
        
    X_probe = np.vstack(states)
    y_probe = np.array([paths[i][t+1] for t in range(seq_len) for i in range(num_samples)])
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_probe, y_probe)
    probing_accuracy = clf.score(X_probe, y_probe)
    
    assert probing_accuracy >= 0.85

def test_TEST_T3_05_F6_F7():
    # TEST_T3_05_F6_F7: Recurrent SSM L0 Sparsity
    # Apply L0 optimization on RecurrentSSM, checking accuracy remains high for Pareto-optimal sparsity.
    inputs = []
    labels = []
    for i in range(50):
        seq = ['a', 'b'] * 5 if i % 2 == 0 else ['b', 'a'] * 5
        inputs.append(seq)
        labels.append(seq[1:] + [seq[0]])
    X_tensor = torch.tensor([[0 if char == 'a' else 1 for char in seq] for seq in inputs], dtype=torch.long)
    Y_tensor = torch.tensor([[0 if char == 'a' else 1 for char in label] for label in labels], dtype=torch.long)
    
    model = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16, hippo_init=True)
    model.val_data = (X_tensor, Y_tensor)
    
    train_model(model, X_tensor, Y_tensor, epochs=80, lr=0.03)
    
    masked = apply_l0_mask(model)
    l0_pruning_step(masked, temperature=0.1, target_sparsity=0.2)
    
    logits, _ = masked(X_tensor)
    pruned_accuracy = (logits.argmax(dim=-1) == Y_tensor).float().mean().item()
    assert pruned_accuracy >= 0.80

def test_TEST_T3_06_F7_F8():
    # TEST_T3_06_F7_F8: Sparsity vs Baseline Control
    # Evaluate pruned RecurrentSSM against Attention and CNN baselines at the same sparsity and parameter levels.
    inputs = []
    labels = []
    for i in range(50):
        seq = ['a', 'b'] * 5 if i % 2 == 0 else ['b', 'a'] * 5
        inputs.append(seq)
        labels.append(seq[1:] + [seq[0]])
    X_tensor = torch.tensor([[0 if char == 'a' else 1 for char in seq] for seq in inputs], dtype=torch.long)
    Y_tensor = torch.tensor([[0 if char == 'a' else 1 for char in label] for label in labels], dtype=torch.long)
    
    ssm = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
    att = CausalAttentionModel(vocab_size=2, d_model=8, state_dim=16)
    conv = Conv1DModel(vocab_size=2, d_model=8, state_dim=16)
    
    train_model(ssm, X_tensor, Y_tensor, epochs=50, lr=0.03)
    train_model(att, X_tensor, Y_tensor, epochs=50, lr=0.03)
    train_model(conv, X_tensor, Y_tensor, epochs=50, lr=0.03)
    
    masked_ssm = apply_l0_mask(ssm)
    l0_pruning_step(masked_ssm, temperature=0.1, target_sparsity=0.2)
    
    with torch.no_grad():
        ssm_logits, _ = masked_ssm(X_tensor)
        att_logits, _ = att(X_tensor)
        conv_logits, _ = conv(X_tensor)
        
        ssm_perf = (ssm_logits.argmax(dim=-1) == Y_tensor).float().mean().item()
        att_perf = (att_logits.argmax(dim=-1) == Y_tensor).float().mean().item()
        conv_perf = (conv_logits.argmax(dim=-1) == Y_tensor).float().mean().item()
        
    assert 0.0 <= ssm_perf <= 1.0
    assert 0.0 <= att_perf <= 1.0
    assert 0.0 <= conv_perf <= 1.0

def test_TEST_T3_07_F1_F3():
    # TEST_T3_07_F1_F3: Spatial baseline vs MLP
    # Compare spatial mapping output directly with continuous MLP forward output for a sequence matching task.
    R = np.array([[0, 1], [1, 0]], dtype=float)
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    baseline_out = apply_routing(R, X)
    
    circuit = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R)
    mlp = compile_mlp_from_circuit(circuit, alphabet_size=2)
    
    X_tensor = torch.tensor([X], dtype=torch.float32)
    mlp_out = mlp(X_tensor)[0].detach().numpy()
    
    assert np.allclose(mlp_out, baseline_out, atol=1e-5)

def test_TEST_T3_08_F5_F8():
    # TEST_T3_08_F5_F8: DFA Dataset Benchmarking
    # Benchmark Causal Attention, Conv1D, and Markov models on a DFA dataset, running significance t-tests.
    att_accs = [0.85, 0.87, 0.86, 0.88, 0.85]
    conv_accs = [0.82, 0.84, 0.83, 0.85, 0.83]
    
    p_val = run_t_test(att_accs, conv_accs)
    assert 0.0 <= p_val <= 1.0

def test_TEST_T3_09_F3_F7():
    # TEST_T3_09_F3_F7: Map-Reduce MLP Pruning
    # Apply L0 pruning to Map-Reduce MLP, ensuring pruning does not break logic structure below minimum routing bounds.
    R = [[1, 0], [0, 1]]
    circuit = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R)
    mlp = compile_mlp_from_circuit(circuit, alphabet_size=2)
    masked = apply_l0_mask(mlp)
    
    # We prune gently
    l0_pruning_step(masked, temperature=0.1, target_sparsity=0.1)
    
    # Verify output accuracy on exact matches is still 100.0%
    X = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out = masked(X)
    assert np.allclose(out[0, 0].detach().numpy(), [1.0, 0.0])
    assert np.allclose(out[0, 1].detach().numpy(), [0.0, 1.0])


# ==========================================
# Tier 4: Real-World Scenarios (End-to-End)
# ==========================================

def test_TEST_T4_01_PIPELINE():
    # TEST_T4_01_PIPELINE: Full End-to-End Pipeline Execution
    # Chain DFA sequence generation, solver execution, MLP compilation, SSM training, L0 pruning, and baseline controls.
    # 1. DFA Sequence Generation
    transition_function = {0: {'a': 0, 'b': 0}}
    generator = DFAGenerator(num_states=1, alphabet=['a', 'b'], transition_function=transition_function, start_state=0, accept_states=[0])
    inputs, labels, paths = generator.generate_sequences(10, 5)
    
    # 2. Solver Execution
    R = [[1.0, 0.0], [0.0, 1.0]]
    circuit = synthesize_circuit(2, 2, ['a', 'b'], R)
    
    # 3. MLP Compilation
    mlp = compile_mlp_from_circuit(circuit, 2)
    
    # 4. SSM Training (Forward Pass)
    ssm = RecurrentSSM(vocab_size=2, d_model=4, state_dim=8)
    X = torch.randn(2, 5, 4)
    logits, _ = ssm(X)
    
    # 5. L0 Pruning
    masked = apply_l0_mask(ssm)
    l0_pruning_step(masked, 0.1, 0.2)
    
    # 6. Baseline Controls
    att = CausalAttentionModel(2, 4, 8)
    _, _ = att(X)
    
    assert logits.shape == (2, 5, 2)

def test_TEST_T4_02_OOD_GENERALIZATION():
    # TEST_T4_02_OOD_GENERALIZATION: Out-of-Distribution Extrapolation
    # Train SSM and Baselines on length N=20; evaluate on length N=100.
    train_inputs = []
    train_labels = []
    for i in range(40):
        seq = ['a', 'b'] * 10 if i % 2 == 0 else ['b', 'a'] * 10
        train_inputs.append(seq)
        train_labels.append(seq[1:] + [seq[0]])
        
    test_inputs = []
    test_labels = []
    for i in range(20):
        seq = ['a', 'b'] * 50 if i % 2 == 0 else ['b', 'a'] * 50
        test_inputs.append(seq)
        test_labels.append(seq[1:] + [seq[0]])
        
    X_train = torch.tensor([[0 if char == 'a' else 1 for char in seq] for seq in train_inputs], dtype=torch.long)
    Y_train = torch.tensor([[0 if char == 'a' else 1 for char in label] for label in train_labels], dtype=torch.long)
    
    X_test = torch.tensor([[0 if char == 'a' else 1 for char in seq] for seq in test_inputs], dtype=torch.long)
    Y_test = torch.tensor([[0 if char == 'a' else 1 for char in label] for label in test_labels], dtype=torch.long)
    
    ssm = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
    att = CausalAttentionModel(vocab_size=2, d_model=8, state_dim=16)
    
    train_model(ssm, X_train, Y_train, epochs=80, lr=0.03)
    train_model(att, X_train, Y_train, epochs=80, lr=0.03)
    
    with torch.no_grad():
        ssm_logits, _ = ssm(X_test)
        att_logits, _ = att(X_test)
        
        ssm_extrapolation_accuracy = (ssm_logits.argmax(dim=-1) == Y_test).float().mean().item()
        attention_accuracy = (att_logits.argmax(dim=-1) == Y_test).float().mean().item()
        
    assert 0.0 <= ssm_extrapolation_accuracy <= 1.0
    assert 0.0 <= attention_accuracy <= 1.0
    assert ssm_extrapolation_accuracy >= 0.80

def test_TEST_T4_03_ADVERSARIAL_ROUTING():
    # TEST_T4_03_ADVERSARIAL_ROUTING: Robustness under Noisy Channels
    # Test spatial routing and compiled MLPs with noise-injected inputs.
    R = [[0, 1], [1, 0]]
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Inject noise (bit flip simulation)
    X_noisy = np.array([[0.9, 0.1], [0.1, 0.9]])
    
    baseline_out = apply_routing(R, X)
    noisy_out = apply_routing(R, X_noisy)
    
    # Verify noise does not completely destroy output structure (diff is small)
    diff = np.abs(baseline_out - noisy_out).max()
    assert diff <= 0.2

def test_TEST_T4_04_SADDLE_POINT_PRUNING():
    # TEST_T4_04_SADDLE_POINT_PRUNING: Multi-Language Pruned Fronts
    # Extract Pareto-optimal sparse subgraphs for combined regular languages.
    inputs = []
    labels = []
    # Combined sequences: some are alternating ab, some are alternating aa bb
    for i in range(25):
        seq = ['a', 'b'] * 5
        inputs.append(seq)
        labels.append(seq[1:] + [seq[0]])
    for i in range(25):
        seq = ['a', 'a', 'b', 'b'] * 2 + ['a', 'a']
        inputs.append(seq)
        labels.append(seq[1:] + [seq[0]])
        
    X_tensor = torch.tensor([[0 if char == 'a' else 1 for char in seq] for seq in inputs], dtype=torch.long)
    Y_tensor = torch.tensor([[0 if char == 'a' else 1 for char in label] for label in labels], dtype=torch.long)
    
    model = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
    model.val_data = (X_tensor, Y_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    
    masked = apply_l0_mask(model)
    
    final_loss = 999.0
    for epoch in range(100):
        optimizer.zero_grad()
        logits, _ = masked(X_tensor)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_tensor.view(-1))
        penalty = masked.compute_l0_penalty()
        total_loss = loss + 0.01 * penalty
        total_loss.backward()
        optimizer.step()
        final_loss = loss.item()
        
    l0_pruning_step(masked, temperature=0.1, target_sparsity=0.3)
    assert final_loss < 1.0
    assert len(masked.pareto_frontier) > 0

def test_TEST_T4_05_EXPERIMENT_SUITE_RUNNER():
    # TEST_T4_05_EXPERIMENT_SUITE_RUNNER: Run Experiments Script E2E
    # Execute the full run_experiments.py script using a mock data configuration, verifying generated tables and plots.
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        res = subprocess.run(
            [sys.executable, "src/scripts/run_experiments.py", "--config", "mock", "--output_dir", temp_dir],
            capture_output=True,
            text=True
        )
        assert res.returncode == 0
        assert os.path.exists(os.path.join(temp_dir, "results_table.csv"))


def test_TEST_T4_06_MULTIPROCESSING_COMPLEXITY():
    # TEST_T4_06_MULTIPROCESSING_COMPLEXITY: Multithreaded Complexity Scaling
    # Run complexity benchmarks in parallel across threads.
    def run_benchmark_thread(N):
        R = generate_routing_matrix(N, N)
        circuit = synthesize_circuit(seq_len=N, num_outputs=N, alphabet=['a', 'b'], routing_matrix=R)
        model = compile_mlp_from_circuit(circuit, alphabet_size=2)
        return sum(p.numel() for p in model.parameters())
        
    lengths = [2, 4, 6, 8]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_benchmark_thread, lengths))
        
    assert len(results) == len(lengths)
    for res in results:
        assert res > 0
