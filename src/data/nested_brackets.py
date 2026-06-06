import random
from typing import Tuple
import torch

def generate_nested_brackets(
    num_samples: int,
    length: int = None,
    depth: int = None,
    num_bracket_types: int = 1,
    seq_len: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates perfectly balanced, deeply nested sequences of n different bracket types.
    
    Even integers 2*k represent open brackets, and odd integers 2*k + 1 represent close brackets,
    where k ranges from 0 to num_bracket_types - 1.
    
    Parameters:
    -----------
    num_samples: int
        Number of sequences to generate.
    length: int, optional
        Length of each sequence (must be even).
    depth: int
        Maximum nesting depth (must be <= length // 2).
    num_bracket_types: int, default 1
        Number of different bracket types.
    seq_len: int, optional
        Alias for length, used for backward compatibility.
    
    Returns:
    --------
    inputs: torch.Tensor
        Input sequences of shape (num_samples, length) of type torch.long.
    targets: torch.Tensor
        Target sequences of shape (num_samples, length) of type torch.long, where targets
        contains the left-shifted inputs, with the last step set to 0.
    """
    # Handle seq_len / length aliases
    actual_length = length if seq_len is None else seq_len
    
    # Validation
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if actual_length is None:
        raise ValueError("length or seq_len must be specified")
    if actual_length <= 0:
        raise ValueError("length must be positive")
    if actual_length % 2 != 0:
        raise ValueError("length must be even")
    if depth is None:
        raise ValueError("depth must be specified")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth > actual_length // 2:
        raise ValueError("depth cannot exceed length // 2")
    if num_bracket_types <= 0:
        raise ValueError("num_bracket_types must be positive")

    inputs_list = []
    targets_list = []

    for _ in range(num_samples):
        seq = []
        stack = []
        max_d = 0
        for t in range(actual_length):
            C = len(stack)
            R = actual_length - t
            
            # Constraints
            must_open = (C == 0) or (max_d < depth and C + R <= 2 * depth)
            must_close = (C == R) or (C == depth)
            
            if must_open:
                action = "open"
            elif must_close:
                action = "close"
            else:
                action = "open" if random.random() < 0.5 else "close"
                
            if action == "open":
                k = random.randint(0, num_bracket_types - 1)
                token = 2 * k
                seq.append(token)
                stack.append(k)
                max_d = max(max_d, len(stack))
            else:
                k = stack.pop()
                token = 2 * k + 1
                seq.append(token)
        
        inputs_list.append(seq)
        
        # Target is left-shifted input, with the last token set to 0 (first token of vocab)
        target = seq[1:] + [0]
        targets_list.append(target)

    inputs_tensor = torch.tensor(inputs_list, dtype=torch.long)
    targets_tensor = torch.tensor(targets_list, dtype=torch.long)
    
    return inputs_tensor, targets_tensor
