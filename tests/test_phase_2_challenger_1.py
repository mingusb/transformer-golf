import pytest
import torch
import torch.nn as nn
import numpy as np
import multiprocessing
from typing import Tuple

from src.data.nested_brackets import generate_nested_brackets
from src.models.stack_rnn import StackRNN
from src.models.lsm import LiquidStateMachine

# =====================================================================
# 1. Nested Brackets Generator Adversarial & Boundary Tests
# =====================================================================

def test_nested_brackets_alias_precedence():
    """Verify that seq_len takes precedence over length and both work."""
    inputs_seq, targets_seq = generate_nested_brackets(num_samples=5, seq_len=10, depth=3, num_bracket_types=1)
    inputs_len, targets_len = generate_nested_brackets(num_samples=5, length=10, depth=3, num_bracket_types=1)
    assert inputs_seq.shape == (5, 10)
    assert inputs_len.shape == (5, 10)
    
    # If both are specified, seq_len takes precedence
    inputs_both, _ = generate_nested_brackets(num_samples=5, length=10, seq_len=14, depth=3, num_bracket_types=1)
    assert inputs_both.shape == (5, 14)

def test_nested_brackets_extreme_vocab():
    """Verify nested brackets generator with extreme bracket type counts."""
    num_bracket_types = 5000
    inputs, targets = generate_nested_brackets(num_samples=2, length=10, depth=3, num_bracket_types=num_bracket_types)
    assert inputs.shape == (2, 10)
    # Open brackets are even, closed brackets are odd
    for seq in inputs.tolist():
        for token in seq:
            assert 0 <= token < 2 * num_bracket_types

def test_nested_brackets_boundary_depth():
    """Verify nested brackets depth edge conditions."""
    # Depth must be <= length // 2
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=10, depth=6)
        
    # Minimum depth/length configuration
    inputs, targets = generate_nested_brackets(num_samples=5, length=2, depth=1, num_bracket_types=1)
    assert inputs.shape == (5, 2)
    # The only balanced sequence of length 2 is open (0), close (1)
    for seq in inputs.tolist():
        assert seq == [0, 1]

# =====================================================================
# 2. StackRNN Adversarial & Boundary Tests
# =====================================================================

def test_stack_rnn_empty_seq():
    """Verify StackRNN behaves correctly when sequence length is 0."""
    vocab_size = 4
    hidden_size = 8
    stack_width = 2
    stack_depth = 3
    model = StackRNN(vocab_size=vocab_size, hidden_size=hidden_size, stack_width=stack_width, stack_depth=stack_depth)
    
    x = torch.zeros((2, 0), dtype=torch.long)
    logits, stack_states = model(x)
    
    # Expected output shapes for empty sequence
    assert logits.shape == (2, 0, vocab_size)
    assert stack_states.shape == (2, 0, stack_depth, stack_width)

def test_stack_rnn_state_1d_crash():
    """Verify StackRNN raises a descriptive RuntimeError when a 1D state is passed.
    Instead, it crashes with IndexError if batch_size == hidden_size."""
    vocab_size = 4
    hidden_size = 8
    model = StackRNN(vocab_size=vocab_size, hidden_size=hidden_size, stack_width=2, stack_depth=3)
    
    # When batch_size == hidden_size, state.shape[0] == batch_size,
    # so state.shape[1] is evaluated, throwing IndexError instead of a clean error.
    x = torch.randint(0, vocab_size, (hidden_size, 5))
    state_1d = torch.zeros(hidden_size)
    
    try:
        model(x, state=state_1d)
    except RuntimeError as e:
        # If the model handles it or if it evaluates to RuntimeError
        assert "State shape size mismatch" in str(e)
    except IndexError as e:
        pytest.fail(f"Bug exposed: StackRNN raised IndexError instead of RuntimeError when batch_size == hidden_size. Error: {e}")

def test_stack_rnn_invalid_token_bounds():
    """Verify StackRNN raises IndexError for out-of-vocabulary tokens."""
    vocab_size = 4
    model = StackRNN(vocab_size=vocab_size, hidden_size=8, stack_width=2, stack_depth=3)
    
    x_out_of_bounds = torch.tensor([[0, 4, 1]], dtype=torch.long)  # 4 is out of bounds
    with pytest.raises(IndexError):
        model(x_out_of_bounds)
        
    x_negative = torch.tensor([[0, -1, 1]], dtype=torch.long)  # -1 is out of bounds
    with pytest.raises(IndexError):
        model(x_negative)

# =====================================================================
# 3. Liquid State Machine (LSM) Adversarial & Boundary Tests
# =====================================================================

def run_lsm_init_with_high_sparsity():
    """Target function to run in a separate process for timeout testing."""
    # This constructor will loop infinitely because sparsity=1.0 forces W_res to be all zeros,
    # leading to zero eigenvalues, which keeps the 'while True' loop running forever.
    LiquidStateMachine(input_size=2, reservoir_size=5, output_size=2, sparsity=1.0)

def test_lsm_sparsity_infinite_loop():
    """Expose the infinite loop bug when sparsity is 1.0 (or high enough to zero out W_res)."""
    p = multiprocessing.Process(target=run_lsm_init_with_high_sparsity)
    p.start()
    p.join(timeout=2.0)  # Wait for 2 seconds
    
    if p.is_alive():
        p.terminate()
        p.join()
        pytest.fail("LiquidStateMachine.__init__ timed out! Infinite loop occurred when sparsity=1.0.")

def test_lsm_near_zero_eigenvalues_instability(monkeypatch):
    """Expose that when reservoir eigenvalues are tiny but above 1e-5, weights explode."""
    # Mock torch.randn to return a diagonal matrix with diagonal elements = 1.1e-5
    # The eigenvalues will be 1.1e-5, which is > 1e-5, so the loop terminates.
    def mock_randn(*args, **kwargs):
        return torch.eye(2) * 1.1e-5
    monkeypatch.setattr(torch, "randn", mock_randn)
    
    # Initialize with spectral_radius=0.95.
    # W_res is scaled by (0.95 / 1.1e-5) = 86363.63
    model = LiquidStateMachine(input_size=2, reservoir_size=2, output_size=2, spectral_radius=0.95, sparsity=0.0)
    
    # The resulting W_res has weights scaled up to 0.95
    assert torch.allclose(torch.abs(model.W_res), torch.eye(2) * 0.95)
    
    # What if the random matrix has an off-diagonal element that makes it highly non-normal?
    # e.g., [[1.1e-5, 100], [0, 1.1e-5]]. Eigenvalues are 1.1e-5, but the upper-right element is 100.
    def mock_randn_non_normal(*args, **kwargs):
        m = torch.eye(2) * 1.1e-5
        m[0, 1] = 100.0
        return m
    monkeypatch.setattr(torch, "randn", mock_randn_non_normal)
    
    model_non_normal = LiquidStateMachine(input_size=2, reservoir_size=2, output_size=2, spectral_radius=0.95, sparsity=0.0)
    
    # The upper right element should blow up to 100 * (0.95 / 1.1e-5) ≈ 8.636 million!
    max_weight = torch.max(torch.abs(model_non_normal.W_res)).item()
    assert max_weight > 8.0e6, f"Expected weight explosion, got max weight {max_weight}"

def test_lsm_input_dimensions_mismatch():
    """Verify that dimension mismatches raise proper exceptions."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    
    # 3D float tensor with incorrect feature dimension
    x_bad_features = torch.randn(2, 5, 4)  # input_size is 3, but features dim is 4
    with pytest.raises(ValueError) as exc_info:
        model(x_bad_features)
    assert "does not match input_size" in str(exc_info.value)
    
    # 4D tensor (unsupported dimension)
    x_4d = torch.randn(2, 5, 3, 1)
    with pytest.raises(ValueError) as exc_info:
        model(x_4d)
    assert "Unsupported input dimension" in str(exc_info.value)

def test_lsm_state_mismatches():
    """Verify that invalid state tensors raise appropriate ValueErrors."""
    model = LiquidStateMachine(input_size=2, reservoir_size=10, output_size=2)
    x = torch.randn(2, 5, 2)
    
    # Wrong state batch size
    bad_batch_state = torch.zeros(3, 10)
    with pytest.raises(ValueError) as exc_info:
        model(x, state=bad_batch_state)
    assert "State shape size mismatch" in str(exc_info.value)
    
    # Wrong state reservoir size
    bad_size_state = torch.zeros(2, 9)
    with pytest.raises(ValueError) as exc_info:
        model(x, state=bad_size_state)
    assert "State shape size mismatch" in str(exc_info.value)

# =====================================================================
# 4. Experiments Script CLI Tests
# =====================================================================

import subprocess
import sys
import os

def test_experiments_cli_stack_rnn_alternating():
    """Verify that run_experiments.py runs the alternating task with StackRNN successfully."""
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--config", "mock",
        "--task", "alternating",
        "--model", "stack_rnn",
        "--output_dir", "results/test_challenger_stack_rnn_alt"
    ], capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed with error: {res.stderr}"
    assert os.path.exists("results/test_challenger_stack_rnn_alt/results_table.csv")

def test_experiments_cli_stack_rnn_nesting():
    """Verify that run_experiments.py runs the nesting task with StackRNN successfully."""
    res = subprocess.run([
        sys.executable, "src/scripts/run_experiments.py",
        "--config", "mock",
        "--task", "nesting",
        "--model", "stack_rnn",
        "--output_dir", "results/test_challenger_stack_rnn_nest"
    ], capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed with error: {res.stderr}"
    assert os.path.exists("results/test_challenger_stack_rnn_nest/results_table.csv")
