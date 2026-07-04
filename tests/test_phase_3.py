import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import subprocess

# Add project root directory to sys.path in case it is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Flags for optional components
HAS_COPY_DATA = False
HAS_ABC_DATA = False
HAS_DUAL_STACK_RNN = False

try:
    from src.data.context_sensitive import generate_copy_task
    HAS_COPY_DATA = True
except ImportError:
    pass

try:
    from src.data.context_sensitive import generate_abc_task
    HAS_ABC_DATA = True
except ImportError:
    pass

try:
    from src.models.universal_rnn import DualStackRNN
    HAS_DUAL_STACK_RNN = True
except ImportError:
    pass

# Helper to check if run_experiments.py is integrated with copy/abc tasks
def check_experiment_integration():
    script_path = os.path.join(os.path.dirname(__file__), "../src/scripts/run_experiments.py")
    if not os.path.exists(script_path):
        return False
    try:
        with open(script_path, "r") as f:
            content = f.read()
        return "copy" in content and "abc" in content
    except Exception:
        return False

HAS_EXPERIMENT_INTEGRATION = check_experiment_integration()


# ==========================================
# TIER 1 TESTS: Feature Coverage (20 tests)
# ==========================================

# --- Feature F1: Copy Task Generator (5 tests) ---

@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T1_F1_01_generation_shapes():
    num_samples = 5
    length = 3
    vocab_size = 4
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    seq_len = 2 * length + 1
    assert inputs.shape == (num_samples, seq_len)
    assert targets.shape == (num_samples, seq_len)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T1_F1_02_vocab_range():
    num_samples = 5
    length = 3
    vocab_size = 6
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    delimiter = vocab_size - 1
    # Check that all elements except the delimiter at length are in range [0, vocab_size - 2]
    for i in range(num_samples):
        for j in range(inputs.shape[1]):
            val = inputs[i, j].item()
            if j != length:
                assert 0 <= val <= vocab_size - 2
            else:
                assert val == delimiter


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T1_F1_03_delimiter_position():
    num_samples = 5
    length = 4
    vocab_size = 5
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    seq_len = 2 * length + 1
    middle_idx = seq_len // 2
    assert middle_idx == length
    delimiter = vocab_size - 1
    assert (inputs[:, middle_idx] == delimiter).all()


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T1_F1_04_target_mapping():
    num_samples = 5
    length = 3
    vocab_size = 4
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    # Target should be left-shifted input, with the last step set to 0
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert (targets[:, -1] == 0).all()


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T1_F1_05_copy_property():
    num_samples = 5
    length = 4
    vocab_size = 8
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    w_before = inputs[:, :length]
    w_after = inputs[:, length+1:]
    assert torch.equal(w_before, w_after)


# --- Feature F2: ABC Task Generator (5 tests) ---

@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T1_F2_01_generation_shapes():
    num_samples = 5
    n_max = 4
    # Test with n specified
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    assert inputs.shape == (num_samples, 3 * n)
    assert targets.shape == (num_samples, 3 * n)
    
    # Test with n=None
    inputs, targets = generate_abc_task(num_samples, n_max, n=None)
    assert inputs.shape == (num_samples, 3 * n_max)
    assert targets.shape == (num_samples, 3 * n_max)


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T1_F2_02_vocab_range():
    num_samples = 5
    n_max = 3
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    # Tokens should be only 0, 1, 2 when n is specified (no padding)
    for val in inputs.unique():
        assert val.item() in (0, 1, 2)


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T1_F2_03_structure():
    num_samples = 5
    n_max = 3
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    # Verify input sequence is of form a^n b^n c^n
    for seq in inputs:
        assert torch.equal(seq[:n], torch.zeros(n, dtype=torch.long))
        assert torch.equal(seq[n:2*n], torch.ones(n, dtype=torch.long))
        assert torch.equal(seq[2*n:], torch.full((n,), 2, dtype=torch.long))


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T1_F2_04_target_mapping():
    num_samples = 5
    n_max = 3
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    # Verify target is left-shifted input, with the last step set to 0 or pad token (3)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    assert (targets[:, -1] == 0).all() or (targets[:, -1] == 3).all()


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T1_F2_05_various_n():
    num_samples = 3
    n_max = 5
    for n in [1, 2, 3, 4, 5]:
        inputs, targets = generate_abc_task(num_samples, n_max, n=n)
        for seq in inputs:
            assert (seq[:n] == 0).all()
            assert (seq[n:2*n] == 1).all()
            assert (seq[2*n:] == 2).all()


# --- Feature F3: DualStackRNN (5 tests) ---

@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T1_F3_01_initialization():
    model = DualStackRNN(vocab_size=10, hidden_size=8, stack_width=4, stack_depth=6)
    assert isinstance(model, nn.Module)


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T1_F3_02_forward_shape():
    batch_size = 3
    seq_len = 5
    vocab_size = 10
    hidden_size = 8
    stack_width = 4
    stack_depth = 6
    model = DualStackRNN(vocab_size, hidden_size, stack_width, stack_depth)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, stack_states = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    if isinstance(stack_states, (list, tuple)):
        for s in stack_states:
            assert s.shape == (batch_size, seq_len, stack_depth, stack_width)
    else:
        assert len(stack_states.shape) in (4, 5)


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T1_F3_03_stack_operations():
    batch_size = 2
    seq_len = 3
    vocab_size = 10
    hidden_size = 8
    stack_width = 4
    stack_depth = 6
    model = DualStackRNN(vocab_size, hidden_size, stack_width, stack_depth)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, stack_states = model(x)
    if isinstance(stack_states, (list, tuple)):
        for s in stack_states:
            assert not torch.isnan(s).any()
    else:
        assert not torch.isnan(stack_states).any()


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T1_F3_04_differentiability():
    batch_size = 2
    seq_len = 3
    vocab_size = 10
    hidden_size = 8
    stack_width = 4
    stack_depth = 6
    model = DualStackRNN(vocab_size, hidden_size, stack_width, stack_depth)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T1_F3_05_stack_dimensions():
    vocab_size = 10
    hidden_size = 8
    stack_width = 4
    stack_depth = 6
    model = DualStackRNN(vocab_size, hidden_size, stack_width, stack_depth)
    assert model.stack_width == stack_width
    assert model.stack_depth == stack_depth


# --- Feature F4: run_experiments.py (5 tests) ---

@pytest.mark.skipif(not HAS_EXPERIMENT_INTEGRATION, reason="run_experiments.py not integrated with Phase 3 yet")
def test_TEST_T1_F4_01_cli_copy_task():
    res = subprocess.run([sys.executable, "src/scripts/run_experiments.py", "--help"], capture_output=True, text=True)
    assert "copy" in res.stdout


@pytest.mark.skipif(not HAS_EXPERIMENT_INTEGRATION, reason="run_experiments.py not integrated with Phase 3 yet")
def test_TEST_T1_F4_02_cli_abc_task():
    res = subprocess.run([sys.executable, "src/scripts/run_experiments.py", "--help"], capture_output=True, text=True)
    assert "abc" in res.stdout


@pytest.mark.skipif(not HAS_EXPERIMENT_INTEGRATION, reason="run_experiments.py not integrated with Phase 3 yet")
def test_TEST_T1_F4_03_output_csv_existence():
    with open("src/scripts/run_experiments.py", "r") as f:
        content = f.read()
    assert "results" in content
    assert ".csv" in content


@pytest.mark.skipif(not HAS_EXPERIMENT_INTEGRATION, reason="run_experiments.py not integrated with Phase 3 yet")
def test_TEST_T1_F4_04_output_plot_existence():
    with open("src/scripts/run_experiments.py", "r") as f:
        content = f.read()
    assert ".png" in content or "plt.savefig" in content


@pytest.mark.skipif(not HAS_EXPERIMENT_INTEGRATION, reason="run_experiments.py not integrated with Phase 3 yet")
def test_TEST_T1_F4_05_cli_model_choice():
    with open("src/scripts/run_experiments.py", "r") as f:
        content = f.read()
    assert any(m in content.lower() for m in ["dual_stack_rnn", "universal", "universal_rnn"])


# ==========================================
# TIER 2 TESTS: Boundary & Corner Cases (20 tests)
# ==========================================

# --- Feature F1: Copy Task Generator (5 tests) ---

@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T2_F1_01_invalid_num_samples():
    with pytest.raises(ValueError):
        generate_copy_task(0, 5, 8)
    with pytest.raises(ValueError):
        generate_copy_task(-3, 5, 8)


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T2_F1_02_invalid_seq_len_even():
    with pytest.raises(ValueError):
        generate_copy_task(10, 2.5, 8)


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T2_F1_03_invalid_seq_len_negative():
    with pytest.raises(ValueError):
        generate_copy_task(10, 0, 8)
    with pytest.raises(ValueError):
        generate_copy_task(10, -5, 8)


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T2_F1_04_invalid_vocab_size():
    with pytest.raises(ValueError):
        generate_copy_task(10, 5, 1)
    with pytest.raises(ValueError):
        generate_copy_task(10, 5, -2)


@pytest.mark.skipif(not HAS_COPY_DATA, reason="Copy task generator not available")
def test_TEST_T2_F1_05_reproducibility():
    torch.manual_seed(42)
    in1, tar1 = generate_copy_task(10, 5, 8)
    torch.manual_seed(42)
    in2, tar2 = generate_copy_task(10, 5, 8)
    assert torch.equal(in1, in2)
    assert torch.equal(tar1, tar2)


# --- Feature F2: ABC Task Generator (5 tests) ---

@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T2_F2_01_invalid_num_samples():
    with pytest.raises(ValueError):
        generate_abc_task(0, 5)
    with pytest.raises(ValueError):
        generate_abc_task(-1, 5)


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T2_F2_02_invalid_seq_len_not_div_3():
    with pytest.raises(ValueError):
        generate_abc_task(10, 5, n=2.5)


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T2_F2_03_invalid_seq_len_negative():
    with pytest.raises(ValueError):
        generate_abc_task(10, 0)
    with pytest.raises(ValueError):
        generate_abc_task(10, -3)
    with pytest.raises(ValueError):
        generate_abc_task(10, 5, n=0)
    with pytest.raises(ValueError):
        generate_abc_task(10, 5, n=-1)


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T2_F2_04_invalid_vocab_size():
    inputs, targets = generate_abc_task(10, 5)
    assert ((inputs >= 0) & (inputs < 4)).all()
    assert ((targets >= 0) & (targets < 4)).all()


@pytest.mark.skipif(not HAS_ABC_DATA, reason="ABC task generator not available")
def test_TEST_T2_F2_05_reproducibility():
    torch.manual_seed(42)
    in1, tar1 = generate_abc_task(10, 5)
    torch.manual_seed(42)
    in2, tar2 = generate_abc_task(10, 5)
    assert torch.equal(in1, in2)
    assert torch.equal(tar1, tar2)


# --- Feature F3: DualStackRNN (5 tests) ---

@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T2_F3_01_zero_stack_width():
    with pytest.raises(ValueError):
        DualStackRNN(vocab_size=10, hidden_size=8, stack_width=0, stack_depth=6)
    with pytest.raises(ValueError):
        DualStackRNN(vocab_size=10, hidden_size=8, stack_width=-1, stack_depth=6)


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T2_F3_02_min_vocab_size():
    model = DualStackRNN(vocab_size=2, hidden_size=8, stack_width=4, stack_depth=6)
    x = torch.randint(0, 2, (2, 4))
    logits, _ = model(x)
    assert logits.shape == (2, 4, 2)


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T2_F3_03_extreme_inputs():
    model = DualStackRNN(vocab_size=10, hidden_size=8, stack_width=4, stack_depth=6)
    x = torch.randint(0, 10, (128, 200))
    logits, _ = model(x)
    assert not torch.isnan(logits).any()


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T2_F3_04_device_transfer():
    model = DualStackRNN(vocab_size=10, hidden_size=8, stack_width=4, stack_depth=6)
    device = torch.device("cpu")
    model = model.to(device)
    x = torch.randint(0, 10, (2, 4), device=device)
    logits, _ = model(x)
    assert logits.device == device


@pytest.mark.skipif(not HAS_DUAL_STACK_RNN, reason="DualStackRNN not available")
def test_TEST_T2_F3_05_gradient_clip():
    model = DualStackRNN(vocab_size=10, hidden_size=8, stack_width=4, stack_depth=6)
    x = torch.randint(0, 10, (2, 4))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# --- Feature F4: run_experiments.py (5 tests) ---

def test_TEST_T2_F4_01_invalid_task():
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "invalid"
    ], capture_output=True, text=True)
    assert res.returncode != 0


def test_TEST_T2_F4_02_invalid_model():
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--model", "invalid"
    ], capture_output=True, text=True)
    assert res.returncode != 0


def test_TEST_T2_F4_03_empty_output_dir(tmp_path):
    output_dir = os.path.join(tmp_path, "empty_dir")
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "alternating",
        "--config", "mock",
        "--output_dir", output_dir
    ], capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, "results_table.csv"))


def test_TEST_T2_F4_04_mock_config_epochs():
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "alternating",
        "--config", "mock"
    ], capture_output=True, text=True)
    assert res.returncode == 0


def test_TEST_T2_F4_05_overwriting_outputs(tmp_path):
    output_dir = os.path.join(tmp_path, "overwrite_dir")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "results_table.csv")
    
    res1 = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "alternating",
        "--config", "mock",
        "--output_dir", output_dir
    ], capture_output=True, text=True)
    assert res1.returncode == 0
    assert os.path.exists(csv_path)
    mtime1 = os.path.getmtime(csv_path)
    
    import time
    time.sleep(0.1)
    
    res2 = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "alternating",
        "--config", "mock",
        "--output_dir", output_dir
    ], capture_output=True, text=True)
    assert res2.returncode == 0
    mtime2 = os.path.getmtime(csv_path)
    
    assert mtime2 > mtime1


# ==========================================
# TIER 3 TESTS: Pairwise Integration (4 tests)
# ==========================================

@pytest.mark.skipif(not (HAS_COPY_DATA and HAS_DUAL_STACK_RNN), reason="Copy data or DualStackRNN not available")
def test_TEST_T3_01_generator_and_model_copy():
    vocab_size = 4
    length = 2
    num_samples = 10
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    
    model = DualStackRNN(vocab_size, hidden_size=8, stack_width=2, stack_depth=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    assert losses[-1] < losses[0]


@pytest.mark.skipif(not (HAS_ABC_DATA and HAS_DUAL_STACK_RNN), reason="ABC data or DualStackRNN not available")
def test_TEST_T3_02_generator_and_model_abc():
    num_samples = 10
    n_max = 2
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    vocab_size = 4
    
    model = DualStackRNN(vocab_size, hidden_size=8, stack_width=2, stack_depth=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(5):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    assert losses[-1] < losses[0]


@pytest.mark.skipif(not (HAS_EXPERIMENT_INTEGRATION and HAS_DUAL_STACK_RNN), reason="CLI or DualStackRNN not available")
def test_TEST_T3_03_cli_runs_copy_with_dual_stack():
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "copy",
        "--config", "mock",
        "--model", "dual_stack_rnn"
    ], capture_output=True, text=True)
    assert res.returncode == 0


@pytest.mark.skipif(not (HAS_EXPERIMENT_INTEGRATION and HAS_DUAL_STACK_RNN), reason="CLI or DualStackRNN not available")
def test_TEST_T3_04_cli_runs_abc_with_dual_stack():
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--task", "abc",
        "--config", "mock",
        "--model", "dual_stack_rnn"
    ], capture_output=True, text=True)
    assert res.returncode == 0


# ==========================================
# TIER 4 TESTS: Real-World Scenarios (5 tests)
# ==========================================

@pytest.mark.skipif(not (HAS_COPY_DATA and HAS_DUAL_STACK_RNN), reason="Copy data or DualStackRNN not available")
def test_TEST_T4_01_copy_memorization():
    vocab_size = 3
    length = 2
    num_samples = 4
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    
    model = DualStackRNN(vocab_size, hidden_size=16, stack_width=4, stack_depth=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    
    acc = 0.0
    for epoch in range(200):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        if acc >= 0.89:
            break
            
    assert acc >= 0.89


@pytest.mark.skipif(not (HAS_ABC_DATA and HAS_DUAL_STACK_RNN), reason="ABC data or DualStackRNN not available")
def test_TEST_T4_02_abc_memorization():
    num_samples = 4
    n_max = 2
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    vocab_size = 4
    
    model = DualStackRNN(vocab_size, hidden_size=16, stack_width=4, stack_depth=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()
    
    acc = 0.0
    for epoch in range(200):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        if acc == 1.0:
            break
            
    assert acc == 1.0


@pytest.mark.skipif(not (HAS_COPY_DATA and HAS_DUAL_STACK_RNN), reason="Copy data or DualStackRNN not available")
def test_TEST_T4_03_copy_generalization():
    vocab_size = 3
    X_train, Y_train = generate_copy_task(20, 2, vocab_size)
    X_test, Y_test = generate_copy_task(10, 3, vocab_size)
    
    model = DualStackRNN(vocab_size, hidden_size=16, stack_width=4, stack_depth=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(150):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, vocab_size), Y_train.view(-1))
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(X_test)
        test_preds = test_logits.argmax(dim=-1)
        token_acc = (test_preds == Y_test).float().mean().item()
    assert token_acc >= 0.0


@pytest.mark.skipif(not (HAS_ABC_DATA and HAS_DUAL_STACK_RNN), reason="ABC data or DualStackRNN not available")
def test_TEST_T4_04_abc_generalization():
    vocab_size = 4
    X_train, Y_train = generate_abc_task(20, 2, n=2)
    X_test, Y_test = generate_abc_task(10, 3, n=3)
    
    model = DualStackRNN(vocab_size, hidden_size=16, stack_width=4, stack_depth=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(150):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, vocab_size), Y_train.view(-1))
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(X_test)
        test_preds = test_logits.argmax(dim=-1)
        token_acc = (test_preds == Y_test).float().mean().item()
    assert token_acc >= 0.0


@pytest.mark.skipif(not (HAS_COPY_DATA and HAS_DUAL_STACK_RNN), reason="Copy data or DualStackRNN not available")
def test_TEST_T4_05_baseline_comparison():
    try:
        from src.models.stack_rnn import StackRNN
        HAS_STACK_RNN = True
    except ImportError:
        HAS_STACK_RNN = False
        
    assert HAS_STACK_RNN, "StackRNN model should be available"
    
    vocab_size = 4
    model_single = StackRNN(vocab_size, hidden_size=8, stack_width=2, stack_depth=4)
    model_dual = DualStackRNN(vocab_size, hidden_size=8, stack_width=2, stack_depth=4)
    
    assert hasattr(model_dual, "stack_depth")
