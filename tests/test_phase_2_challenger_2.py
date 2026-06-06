import pytest
import torch
import torch.nn as nn
import numpy as np
import signal
from src.data.nested_brackets import generate_nested_brackets
from src.models.stack_rnn import StackRNN
from src.models.lsm import LiquidStateMachine

# ==========================================
# Adversarial & Edge Case Tests
# ==========================================

def test_lsm_high_sparsity_infinite_loop():
    """Verify that setting sparsity close to 1.0 causes LSM initialization to hang or fail."""
    def handler(signum, frame):
        raise TimeoutError("LSM initialization timed out (likely infinite loop due to high sparsity)")
        
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(2)  # 2 seconds timeout
    
    try:
        # Sparsity = 1.0 makes W_res all zeros, leading to infinite loop in spectral radius scaling.
        # This test is expected to time out, demonstrating the vulnerability.
        LiquidStateMachine(input_size=2, reservoir_size=10, output_size=2, sparsity=1.0)
        signal.alarm(0)
    except TimeoutError as e:
        pytest.fail(f"LSM hung during initialization: {e}")
    except Exception as e:
        signal.alarm(0)
        # If it raises a clean exception in the future, that is fine.
    finally:
        signal.alarm(0)

def test_stack_rnn_out_of_vocab():
    """Verify StackRNN raises IndexError for out-of-vocab inputs."""
    model = StackRNN(vocab_size=5, hidden_size=8, stack_width=2, stack_depth=3)
    # Token 5 is out of vocab (vocab size is 5, so tokens must be 0-4)
    x = torch.tensor([[1, 2, 5, 3]], dtype=torch.long)
    with pytest.raises(IndexError):
        model(x)
        
    # Negative token
    x_neg = torch.tensor([[1, -1, 2]], dtype=torch.long)
    with pytest.raises(IndexError):
        model(x_neg)

def test_stack_rnn_zero_batch():
    """Verify StackRNN handles batch size of 0."""
    model = StackRNN(vocab_size=5, hidden_size=8, stack_width=2, stack_depth=3)
    x = torch.zeros(0, 10, dtype=torch.long)
    logits, stack_states = model(x)
    assert logits.shape == (0, 10, 5)
    assert stack_states.shape == (0, 10, 3, 2)

def test_stack_rnn_zero_length():
    """Verify StackRNN handles sequence length of 0."""
    model = StackRNN(vocab_size=5, hidden_size=8, stack_width=2, stack_depth=3)
    x = torch.zeros(2, 0, dtype=torch.long)
    logits, stack_states = model(x)
    assert logits.shape == (2, 0, 5)
    assert stack_states.shape == (2, 0, 3, 2)

def test_stack_rnn_mismatched_state():
    """Verify StackRNN raises RuntimeError for invalid state shapes or devices."""
    model = StackRNN(vocab_size=5, hidden_size=8, stack_width=2, stack_depth=3)
    x = torch.zeros(2, 5, dtype=torch.long)
    
    # Mismatched hidden size (expected 8, got 10)
    invalid_state = torch.zeros(2, 10)
    with pytest.raises(RuntimeError):
        model(x, state=invalid_state)
        
    # Mismatched batch size (expected 2, got 3)
    invalid_batch = torch.zeros(3, 8)
    with pytest.raises(RuntimeError):
        model(x, state=invalid_batch)

def test_stack_rnn_numerical_stability():
    """Verify StackRNN soft stack operations don't produce NaNs or Inf under extreme seq len."""
    model = StackRNN(vocab_size=5, hidden_size=8, stack_width=2, stack_depth=3)
    # Long sequence of 1000 tokens
    x = torch.randint(0, 5, (2, 1000))
    logits, stack_states = model(x)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(stack_states).any()
    assert not torch.isinf(stack_states).any()

def test_lsm_unsupported_dimensions():
    """Verify LSM raises ValueError for 1D or 4D inputs."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    # 1D input
    x_1d = torch.randint(0, 3, (5,))
    with pytest.raises(ValueError):
        model(x_1d)
    # 4D input
    x_4d = torch.randn(2, 3, 4, 3)
    with pytest.raises(ValueError):
        model(x_4d)

def test_lsm_mismatched_features():
    """Verify LSM raises ValueError if 3D feature dimension does not match input_size."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    x = torch.randn(2, 5, 4)  # 4 features instead of 3
    with pytest.raises(ValueError):
        model(x)

def test_lsm_zero_batch_zero_length():
    """Verify LSM handles zero-batch and zero-length inputs."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    # Zero batch
    x_zb = torch.randint(0, 3, (0, 5))
    logits, h = model(x_zb)
    assert logits.shape == (0, 5, 2)
    assert h.shape == (0, 10)
    
    # Zero length
    x_zl = torch.randint(0, 3, (2, 0))
    logits_zl, h_zl = model(x_zl)
    assert logits_zl.shape == (2, 0, 2)
    assert h_zl.shape == (2, 10)

def test_lsm_state_validation():
    """Verify LSM raises ValueError for mismatched state shapes."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    x = torch.randint(0, 3, (2, 5))
    
    # Shape mismatch (expected (2, 10), got (2, 5))
    invalid_state = torch.zeros(2, 5)
    with pytest.raises(ValueError):
        model(x, state=invalid_state)
        
    # Shape mismatch (expected (2, 10), got (3, 10))
    invalid_batch = torch.zeros(3, 10)
    with pytest.raises(ValueError):
        model(x, state=invalid_batch)

def test_lsm_out_of_bounds_one_hot():
    """Verify LSM raises error when 2D token inputs are out of bounds."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    x = torch.tensor([[1, 3, 0]], dtype=torch.long)  # Token 3 is out of bounds for input_size 3
    with pytest.raises(Exception):
        model(x)

def test_lsm_nan_propagation():
    """Verify LSM propagates NaNs correctly and handles large values."""
    model = LiquidStateMachine(input_size=3, reservoir_size=10, output_size=2)
    x = torch.randn(2, 5, 3)
    x[0, 2, 1] = float('nan')
    logits, h = model(x)
    # The output at index 2 onwards should be NaN due to recurrences
    assert torch.isnan(logits[0, 2:]).all()

def test_nested_brackets_extreme_depth_length():
    """Verify generate_nested_brackets works under extreme length and depth values."""
    # Maximum possible depth for length 100
    inputs, targets = generate_nested_brackets(num_samples=10, length=100, depth=50, num_bracket_types=5)
    assert inputs.shape == (10, 100)
    assert targets.shape == (10, 100)
    
    # Large num_bracket_types
    inputs_large, targets_large = generate_nested_brackets(num_samples=5, length=20, depth=10, num_bracket_types=1000)
    assert (inputs_large < 2000).all()

def test_model_scalability():
    """Verify models scale to very large hidden and stack/reservoir sizes without crash."""
    # Huge StackRNN
    model_stack = StackRNN(vocab_size=10, hidden_size=256, stack_width=64, stack_depth=200)
    x = torch.randint(0, 10, (2, 5))
    logits, _ = model_stack(x)
    assert logits.shape == (2, 5, 10)
    
    # Huge LSM
    model_lsm = LiquidStateMachine(input_size=10, reservoir_size=500, output_size=5)
    x_lsm = torch.randint(0, 10, (2, 5))
    logits_lsm, _ = model_lsm(x_lsm)
    assert logits_lsm.shape == (2, 5, 5)
