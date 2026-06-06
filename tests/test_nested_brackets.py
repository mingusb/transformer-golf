import pytest
import torch
from src.data.nested_brackets import generate_nested_brackets

def test_generate_nested_brackets_shapes():
    """Test output shapes and types."""
    inputs, targets = generate_nested_brackets(num_samples=15, length=30, depth=5, num_bracket_types=3)
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.shape == (15, 30)
    assert targets.shape == (15, 30)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long

def test_generate_nested_brackets_balance_and_lifo():
    """Test that generated sequences are perfectly balanced and adhere to LIFO stack property."""
    num_bracket_types = 4
    inputs, targets = generate_nested_brackets(num_samples=20, seq_len=40, depth=8, num_bracket_types=num_bracket_types)
    
    # 1. Inputs validation
    for seq in inputs.tolist():
        stack = []
        max_d = 0
        for token in seq:
            assert 0 <= token < 2 * num_bracket_types
            if token % 2 == 0:  # Open
                stack.append(token)
                max_d = max(max_d, len(stack))
            else:  # Close
                assert len(stack) > 0, "Closed bracket on empty stack"
                last_open = stack.pop()
                assert last_open == token - 1, f"Mismatched bracket types: open {last_open}, close {token}"
        assert len(stack) == 0, "Sequence not balanced at the end"
        assert max_d <= 8, f"Exceeded max depth limit: {max_d} > 8"

    # 2. Targets validation (must be left-shifted version of inputs with final token 0)
    # Target shape: (num_samples, length)
    expected_targets = torch.cat([inputs[:, 1:], torch.zeros((inputs.shape[0], 1), dtype=torch.long)], dim=1)
    assert torch.equal(targets, expected_targets)

def test_generate_nested_brackets_depth_limit_achieved():
    """Test that at least some generated sequences reach or the depth constraint is properly checked."""
    depth = 4
    length = 20
    # Let's generate a larger number of samples to ensure we see the depth limit being achieved
    inputs, _ = generate_nested_brackets(num_samples=50, length=length, depth=depth, num_bracket_types=1)
    reached_depth = False
    for seq in inputs.tolist():
        curr_d = 0
        max_d = 0
        for token in seq:
            if token % 2 == 0:
                curr_d += 1
                max_d = max(max_d, curr_d)
            else:
                curr_d -= 1
        assert max_d <= depth
        if max_d == depth:
            reached_depth = True
    assert reached_depth, f"None of the generated sequences reached the exact depth limit of {depth}"

def test_generate_nested_brackets_value_errors():
    """Test that invalid parameters raise ValueError."""
    # Negative/zero samples
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=0, length=10, depth=3)
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=-5, length=10, depth=3)
        
    # Odd sequence length
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=11, depth=3)
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, seq_len=15, depth=3)

    # Negative/zero length
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=0, depth=3)
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=-4, depth=3)

    # Invalid depth
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=10, depth=0)
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=10, depth=-2)
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=10, depth=6)  # depth > length // 2

    # Invalid bracket types
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=10, depth=3, num_bracket_types=0)
    with pytest.raises(ValueError):
        generate_nested_brackets(num_samples=5, length=10, depth=3, num_bracket_types=-1)
