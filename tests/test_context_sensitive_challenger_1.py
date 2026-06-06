import pytest
import torch
import time
import resource
import numpy as np
from src.data.context_sensitive import generate_copy_task, generate_abc_task

def get_max_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# 1. Test Large Batch Scales (num_samples = 50,000)
def test_copy_task_large_batch():
    num_samples = 50000
    length = 10
    vocab_size = 10
    
    start_time = time.perf_counter()
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    elapsed = time.perf_counter() - start_time
    
    assert inputs.shape == (50000, 21)
    assert targets.shape == (50000, 21)
    # Check execution time: copying 50,000 samples should be extremely fast (e.g. < 0.2s)
    assert elapsed < 1.0, f"Copy task large batch took too long: {elapsed:.4f}s"
    
    # Correctness check on first and last sample
    for idx in [0, 49999]:
        seq = inputs[idx]
        assert seq[length].item() == vocab_size - 1
        assert torch.equal(seq[:length], seq[length+1:])

def test_abc_task_large_batch_fixed_n():
    num_samples = 50000
    n_max = 10
    n = 5
    
    start_time = time.perf_counter()
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    elapsed = time.perf_counter() - start_time
    
    assert inputs.shape == (50000, 15)
    assert targets.shape == (50000, 15)
    assert elapsed < 1.0, f"ABC task fixed n large batch took too long: {elapsed:.4f}s"
    
    for idx in [0, 49999]:
        seq = inputs[idx]
        assert torch.equal(seq[0:n], torch.zeros(n, dtype=torch.long))
        assert torch.equal(seq[n:2*n], torch.ones(n, dtype=torch.long))
        assert torch.equal(seq[2*n:3*n], torch.full((n,), 2, dtype=torch.long))

def test_abc_task_large_batch_random_n():
    num_samples = 50000
    n_max = 10
    
    start_time = time.perf_counter()
    inputs, targets = generate_abc_task(num_samples, n_max, n=None)
    elapsed = time.perf_counter() - start_time
    
    assert inputs.shape == (50000, 30)
    assert targets.shape == (50000, 30)
    # Note: Because of python loop (50k iterations), this is expected to be slower.
    # We assert a reasonably generous limit like 5.0s, but we will document it.
    assert elapsed < 5.0, f"ABC task random n large batch took too long: {elapsed:.4f}s"

# 2. Test Large Sequences
def test_copy_task_large_sequence():
    num_samples = 10
    length = 2000
    vocab_size = 1000
    
    start_time = time.perf_counter()
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    elapsed = time.perf_counter() - start_time
    
    assert inputs.shape == (10, 4001)
    assert targets.shape == (10, 4001)
    assert elapsed < 1.0
    
    for idx in range(num_samples):
        seq = inputs[idx]
        assert seq[length].item() == vocab_size - 1
        assert torch.equal(seq[:length], seq[length+1:])

def test_abc_task_large_sequence_fixed():
    num_samples = 10
    n_max = 1000
    n = 1000
    
    start_time = time.perf_counter()
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    elapsed = time.perf_counter() - start_time
    
    assert inputs.shape == (10, 3000)
    assert targets.shape == (10, 3000)
    assert elapsed < 1.0
    
    for idx in range(num_samples):
        seq = inputs[idx]
        assert torch.equal(seq[0:n], torch.zeros(n, dtype=torch.long))
        assert torch.equal(seq[n:2*n], torch.ones(n, dtype=torch.long))
        assert torch.equal(seq[2*n:3*n], torch.full((n,), 2, dtype=torch.long))

def test_abc_task_large_sequence_random():
    num_samples = 10
    n_max = 1000
    
    start_time = time.perf_counter()
    inputs, targets = generate_abc_task(num_samples, n_max, n=None)
    elapsed = time.perf_counter() - start_time
    
    assert inputs.shape == (10, 3000)
    assert targets.shape == (10, 3000)
    assert elapsed < 1.0

# 3. Test Minimal Parameters
def test_copy_task_minimal_params():
    inputs, targets = generate_copy_task(num_samples=1, length=1, vocab_size=2)
    assert inputs.shape == (1, 3)
    assert targets.shape == (1, 3)
    # length=1, vocab_size=2: delimiter is 1. The token in w must be 0.
    # w=[0], delimiter=[1], w=[0] => inputs=[0, 1, 0]
    # targets=[1, 0, 0]
    assert torch.equal(inputs, torch.tensor([[0, 1, 0]], dtype=torch.long))
    assert torch.equal(targets, torch.tensor([[1, 0, 0]], dtype=torch.long))

def test_abc_task_minimal_params_fixed():
    inputs, targets = generate_abc_task(num_samples=1, n_max=1, n=1)
    assert inputs.shape == (1, 3)
    assert targets.shape == (1, 3)
    # n=1: inputs=[0, 1, 2]
    # targets=[1, 2, 3]
    assert torch.equal(inputs, torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert torch.equal(targets, torch.tensor([[1, 2, 3]], dtype=torch.long))

def test_abc_task_minimal_params_random():
    inputs, targets = generate_abc_task(num_samples=1, n_max=1, n=None)
    assert inputs.shape == (1, 3)
    assert targets.shape == (1, 3)
    # n_max=1, n=None: n_val must be 1.
    assert torch.equal(inputs, torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert torch.equal(targets, torch.tensor([[1, 2, 3]], dtype=torch.long))

# 4. Test Invalid Argument Types and Coercion
def test_invalid_argument_types():
    # Numpy int types
    for np_type in [np.int32, np.int64]:
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            generate_copy_task(np_type(10), 5, 8)
        with pytest.raises(ValueError, match="length must be a positive integer"):
            generate_copy_task(10, np_type(5), 8)
        with pytest.raises(ValueError, match="vocab_size must be an integer >= 2"):
            generate_copy_task(10, 5, np_type(8))
            
        with pytest.raises(ValueError, match="num_samples must be a positive integer"):
            generate_abc_task(np_type(10), 5)
        with pytest.raises(ValueError, match="n_max must be a positive integer"):
            generate_abc_task(10, np_type(5))
        with pytest.raises(ValueError, match="n must be an integer satisfying 0 < n <= n_max"):
            generate_abc_task(10, 5, n=np_type(3))

    # Float-like strings
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_copy_task("10", 5, 8)
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_copy_task("10.0", 5, 8)
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_abc_task("10", 5)

    # Booleans (True / False)
    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_copy_task(True, 5, 8)
    with pytest.raises(ValueError, match="length must be a positive integer"):
        generate_copy_task(10, False, 8)
    with pytest.raises(ValueError, match="vocab_size must be an integer >= 2"):
        generate_copy_task(10, 5, True)

    with pytest.raises(ValueError, match="num_samples must be a positive integer"):
        generate_abc_task(True, 5)
    with pytest.raises(ValueError, match="n_max must be a positive integer"):
        generate_abc_task(10, False)
    with pytest.raises(ValueError, match="n must be an integer satisfying 0 < n <= n_max"):
        generate_abc_task(10, 5, n=True)

# 5. Check Memory Leaks / Excessive Allocation
def test_memory_growth_sequential_calls():
    # We call the generator 100 times for moderate sizes and track memory growth.
    # If there's a memory leak (like references being held), memory will grow continuously.
    initial_mem = get_max_rss_kb()
    
    for _ in range(100):
        _, _ = generate_copy_task(1000, 50, 20)
        _, _ = generate_abc_task(1000, 50, n=None)
        
    final_mem = get_max_rss_kb()
    growth_kb = final_mem - initial_mem
    # Growth should not be excessively large (e.g., less than 200MB after garbage collection).
    assert growth_kb < 200 * 1024, f"Excessive memory growth detected: {growth_kb / 1024:.2f} MB"
