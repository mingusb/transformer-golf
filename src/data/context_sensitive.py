import torch
from typing import Tuple

def generate_copy_task(
    num_samples: int,
    length: int,
    vocab_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences for the Copy Task (w # w) where w is a sequence of random tokens
    and '#' is a delimiter token.
    
    Parameters:
    -----------
    num_samples : int
        The number of sequence samples to generate. Must be > 0.
    length : int
        The length of the sub-sequence 'w'. The total sequence length returned is 2 * length + 1.
        Must be > 0.
    vocab_size : int
        The total size of the vocabulary (including the delimiter). Must be >= 2.
        
    Returns:
    --------
    inputs : torch.Tensor
        Sequence tensor of shape (num_samples, 2 * length + 1) of type torch.long.
    targets : torch.Tensor
        Left-shifted input sequences of shape (num_samples, 2 * length + 1) of type torch.long,
        with the last token set to 0.
    """
    # Validation
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("length must be a positive integer")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")

    # w consists of tokens randomly chosen from [0, vocab_size - 2]
    # shape: (num_samples, length)
    w = torch.randint(0, vocab_size - 1, (num_samples, length), dtype=torch.long)
    
    # delimiter has value vocab_size - 1
    # shape: (num_samples, 1)
    delimiter = torch.full((num_samples, 1), vocab_size - 1, dtype=torch.long)
    
    # inputs is: w | delimiter | w
    inputs = torch.cat([w, delimiter, w], dim=1)
    
    # targets is: w[:, 1:] | delimiter | w | padding (0)
    # which is left-shifted inputs, with last token set to 0
    padding = torch.zeros((num_samples, 1), dtype=torch.long)
    targets = torch.cat([inputs[:, 1:], padding], dim=1)
    
    return inputs, targets


def generate_abc_task(
    num_samples: int,
    n_max: int,
    n: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences of the form a^n b^n c^n to test multiple counting.
    
    Parameters:
    -----------
    num_samples : int
        The number of sequence samples to generate. Must be > 0.
    n_max : int
        The maximum count of each character. Must be > 0.
    n : int, optional
        If specified, generates sequences with exactly 'n' of each character.
        If specified, total sequence length is 3 * n. Must satisfy 0 < n <= n_max.
        If None, 'n' is randomly sampled for each sequence from the range [1, n_max],
        and sequences are right-padded to 3 * n_max using the padding token (3).
        
    Returns:
    --------
    inputs : torch.Tensor
        Sequence tensor of shape (num_samples, 3 * n_max) [if n is None] 
        or (num_samples, 3 * n) [if n is specified] of type torch.long.
    targets : torch.Tensor
        Left-shifted input sequences of type torch.long. The last token is set to the padding token (3).
    """
    # Validation
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(n_max, int) or isinstance(n_max, bool) or n_max <= 0:
        raise ValueError("n_max must be a positive integer")
    if n is not None:
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n > n_max:
            raise ValueError("n must be an integer satisfying 0 < n <= n_max")

    if n is not None:
        # Generate sequences of length exactly 3 * n with no padding
        # shape of each sequence is 3 * n
        a = torch.zeros(n, dtype=torch.long)
        b = torch.ones(n, dtype=torch.long)
        c = torch.full((n,), 2, dtype=torch.long)
        single_seq = torch.cat([a, b, c], dim=0)
        inputs = single_seq.unsqueeze(0).repeat(num_samples, 1)
    else:
        # n is randomly sampled for each sequence from range [1, n_max]
        # and sequences are right-padded to 3 * n_max using padding token (3)
        n_sampled = torch.randint(1, n_max + 1, (num_samples,), dtype=torch.long)
        inputs = torch.full((num_samples, 3 * n_max), 3, dtype=torch.long)
        for i in range(num_samples):
            n_val = int(n_sampled[i].item())
            inputs[i, 0 : n_val] = 0
            inputs[i, n_val : 2 * n_val] = 1
            inputs[i, 2 * n_val : 3 * n_val] = 2

    # targets is left-shifted version, with the last token set to 3 (the padding/EOS token)
    padding = torch.full((num_samples, 1), 3, dtype=torch.long)
    targets = torch.cat([inputs[:, 1:], padding], dim=1)

    return inputs, targets
