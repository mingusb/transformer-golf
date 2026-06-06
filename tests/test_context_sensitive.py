import pytest
import torch
from src.data.context_sensitive import generate_copy_task, generate_abc_task

def test_generate_copy_task_shapes_and_dtypes():
    """Test output shapes, dtypes, and tensor types for the Copy task."""
    num_samples = 10
    length = 5
    vocab_size = 8
    
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    
    expected_length = 2 * length + 1
    assert inputs.shape == (num_samples, expected_length)
    assert targets.shape == (num_samples, expected_length)


def test_generate_copy_task_correctness():
    """Verify sequence contents, delimiter position and shifting logic for the Copy task."""
    num_samples = 20
    length = 4
    vocab_size = 6
    delimiter = vocab_size - 1  # 5
    
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    
    # 1. Content checks on inputs
    for seq in inputs:
        # Delimiter must be in the middle: index = length
        assert seq[length].item() == delimiter
        
        # Sub-sequence w before delimiter
        w_before = seq[:length]
        # Sub-sequence w after delimiter
        w_after = seq[length+1:]
        
        assert torch.equal(w_before, w_after)
        
        # All tokens in w must be in [0, vocab_size - 2]
        for token in w_before:
            assert 0 <= token.item() <= vocab_size - 2
            
    # 2. Target checks (left-shifted input, last token is 0)
    expected_targets = torch.cat([inputs[:, 1:], torch.zeros((num_samples, 1), dtype=torch.long)], dim=1)
    assert torch.equal(targets, expected_targets)


def test_generate_copy_task_boundary_vocab():
    """Verify vocab_size = 2 (minimal possible vocab size) works correctly."""
    num_samples = 5
    length = 3
    vocab_size = 2
    delimiter = 1
    
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    
    # w must contain only 0s
    for seq in inputs:
        assert seq[length].item() == delimiter
        assert torch.equal(seq[:length], torch.zeros(length, dtype=torch.long))
        assert torch.equal(seq[length+1:], torch.zeros(length, dtype=torch.long))
        
    expected_targets = torch.cat([inputs[:, 1:], torch.zeros((num_samples, 1), dtype=torch.long)], dim=1)
    assert torch.equal(targets, expected_targets)


def test_generate_copy_task_invalid_params():
    """Verify ValueError is raised for invalid parameters in copy task."""
    # Invalid num_samples
    with pytest.raises(ValueError):
        generate_copy_task(0, 5, 8)
    with pytest.raises(ValueError):
        generate_copy_task(-3, 5, 8)
    with pytest.raises(ValueError):
        generate_copy_task(5.5, 5, 8)  # type checking
    with pytest.raises(ValueError):
        generate_copy_task(True, 5, 8)  # boolean
        
    # Invalid length
    with pytest.raises(ValueError):
        generate_copy_task(10, 0, 8)
    with pytest.raises(ValueError):
        generate_copy_task(10, -2, 8)
    with pytest.raises(ValueError):
        generate_copy_task(10, 5.0, 8)
    with pytest.raises(ValueError):
        generate_copy_task(10, False, 8)

    # Invalid vocab_size
    with pytest.raises(ValueError):
        generate_copy_task(10, 5, 1)
    with pytest.raises(ValueError):
        generate_copy_task(10, 5, 0)
    with pytest.raises(ValueError):
        generate_copy_task(10, 5, -5)
    with pytest.raises(ValueError):
        generate_copy_task(10, 5, 8.5)


def test_generate_abc_task_fixed_n_shapes_and_correctness():
    """Test multiple counting generator with a fixed n."""
    num_samples = 15
    n_max = 6
    n = 4
    
    inputs, targets = generate_abc_task(num_samples, n_max, n)
    
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    
    expected_len = 3 * n
    assert inputs.shape == (num_samples, expected_len)
    assert targets.shape == (num_samples, expected_len)
    
    # Check sequences and contents: a^n b^n c^n
    for seq in inputs:
        assert torch.equal(seq[0:n], torch.zeros(n, dtype=torch.long))
        assert torch.equal(seq[n:2*n], torch.ones(n, dtype=torch.long))
        assert torch.equal(seq[2*n:3*n], torch.full((n,), 2, dtype=torch.long))
        
    # Targets should be left-shifted, last token set to 3 (the pad token)
    expected_targets = torch.cat([inputs[:, 1:], torch.full((num_samples, 1), 3, dtype=torch.long)], dim=1)
    assert torch.equal(targets, expected_targets)


def test_generate_abc_task_variable_n_shapes_and_correctness():
    """Test multiple counting generator with randomly sampled n."""
    num_samples = 30
    n_max = 5
    
    inputs, targets = generate_abc_task(num_samples, n_max, n=None)
    
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    
    expected_len = 3 * n_max
    assert inputs.shape == (num_samples, expected_len)
    assert targets.shape == (num_samples, expected_len)
    
    # Check sequences
    # Each sequence i should have some n_val in [1, n_max]
    # format: 0*n_val + 1*n_val + 2*n_val + 3*(expected_len - 3*n_val)
    for seq in inputs:
        # Determine n_val from number of zeros at the beginning
        # (Since it starts with a run of zeros of length n_val)
        n_val = 0
        while n_val < len(seq) and seq[n_val].item() == 0:
            n_val += 1
            
        assert 1 <= n_val <= n_max, f"Sampled count n_val={n_val} out of bounds"
        
        # Verify the structure: a^n b^n c^n PAD*
        assert torch.equal(seq[0:n_val], torch.zeros(n_val, dtype=torch.long))
        assert torch.equal(seq[n_val:2*n_val], torch.ones(n_val, dtype=torch.long))
        assert torch.equal(seq[2*n_val:3*n_val], torch.full((n_val,), 2, dtype=torch.long))
        assert torch.equal(seq[3*n_val:], torch.full((expected_len - 3*n_val,), 3, dtype=torch.long))
        
    # Targets should be left-shifted, last token set to 3 (the pad token)
    expected_targets = torch.cat([inputs[:, 1:], torch.full((num_samples, 1), 3, dtype=torch.long)], dim=1)
    assert torch.equal(targets, expected_targets)


def test_generate_abc_task_invalid_params():
    """Verify ValueError is raised for invalid parameters in ABC task."""
    # Invalid num_samples
    with pytest.raises(ValueError):
        generate_abc_task(0, 5)
    with pytest.raises(ValueError):
        generate_abc_task(-1, 5)
    with pytest.raises(ValueError):
        generate_abc_task(5.5, 5)
    with pytest.raises(ValueError):
        generate_abc_task(True, 5)

    # Invalid n_max
    with pytest.raises(ValueError):
        generate_abc_task(5, 0)
    with pytest.raises(ValueError):
        generate_abc_task(5, -3)
    with pytest.raises(ValueError):
        generate_abc_task(5, 4.5)
    with pytest.raises(ValueError):
        generate_abc_task(5, False)

    # Invalid n (when specified)
    # n <= 0
    with pytest.raises(ValueError):
        generate_abc_task(5, 5, n=0)
    with pytest.raises(ValueError):
        generate_abc_task(5, 5, n=-2)
    # n > n_max
    with pytest.raises(ValueError):
        generate_abc_task(5, 5, n=6)
    # n not int
    with pytest.raises(ValueError):
        generate_abc_task(5, 5, n=3.5)
    with pytest.raises(ValueError):
        generate_abc_task(5, 5, n=True)
