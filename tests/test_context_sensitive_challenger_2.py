import pytest
import torch
import threading
import concurrent.futures
import tracemalloc
import time
from src.data.context_sensitive import generate_copy_task, generate_abc_task

# -------------------------------------------------------------------------
# Concurrency & Thread-Safety Tests
# -------------------------------------------------------------------------

def run_concurrent_copy_worker(barrier, num_samples, length, vocab_size, thread_id, results):
    """Worker to call generate_copy_task and verify its output correctness."""
    try:
        # Sync all threads to start at the exact same time
        barrier.wait()
        
        inputs, targets = generate_copy_task(num_samples, length, vocab_size)
        
        # Verify shape, dtype, device
        assert inputs.shape == (num_samples, 2 * length + 1)
        assert targets.shape == (num_samples, 2 * length + 1)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
        
        # Verify copy structure: w # w
        delimiter = vocab_size - 1
        for seq in inputs:
            assert seq[length].item() == delimiter
            w_before = seq[:length]
            w_after = seq[length+1:]
            assert torch.equal(w_before, w_after)
            # Ensure tokens are in [0, vocab_size - 2]
            assert torch.all(w_before >= 0)
            assert torch.all(w_before <= vocab_size - 2)
            
        # Verify targets
        expected_targets = torch.cat([inputs[:, 1:], torch.zeros((num_samples, 1), dtype=torch.long)], dim=1)
        assert torch.equal(targets, expected_targets)
        
        results[thread_id] = "OK"
    except Exception as e:
        results[thread_id] = str(e)


def test_concurrency_copy_task():
    """Verify that generate_copy_task is thread-safe under concurrent access."""
    num_threads = 10
    num_samples = 200
    length = 15
    vocab_size = 50
    
    barrier = threading.Barrier(num_threads)
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            futures.append(
                executor.submit(
                    run_concurrent_copy_worker,
                    barrier,
                    num_samples,
                    length,
                    vocab_size,
                    i,
                    results
                )
            )
        concurrent.futures.wait(futures)
        
    for thread_id, status in results.items():
        assert status == "OK", f"Thread {thread_id} failed with error: {status}"


def run_concurrent_abc_worker(barrier, num_samples, n_max, n, thread_id, results):
    """Worker to call generate_abc_task and verify its output correctness."""
    try:
        barrier.wait()
        
        inputs, targets = generate_abc_task(num_samples, n_max, n)
        
        # Verify shapes
        if n is not None:
            expected_length = 3 * n
        else:
            expected_length = 3 * n_max
            
        assert inputs.shape == (num_samples, expected_length)
        assert targets.shape == (num_samples, expected_length)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
        
        # Verify sequences
        if n is not None:
            for seq in inputs:
                assert torch.equal(seq[0:n], torch.zeros(n, dtype=torch.long))
                assert torch.equal(seq[n:2*n], torch.ones(n, dtype=torch.long))
                assert torch.equal(seq[2*n:3*n], torch.full((n,), 2, dtype=torch.long))
        else:
            for seq in inputs:
                # Find length of run of zeros
                n_val = 0
                while n_val < len(seq) and seq[n_val].item() == 0:
                    n_val += 1
                assert 1 <= n_val <= n_max
                # Verify structure
                assert torch.equal(seq[0:n_val], torch.zeros(n_val, dtype=torch.long))
                assert torch.equal(seq[n_val:2*n_val], torch.ones(n_val, dtype=torch.long))
                assert torch.equal(seq[2*n_val:3*n_val], torch.full((n_val,), 2, dtype=torch.long))
                assert torch.equal(seq[3*n_val:], torch.full((expected_length - 3*n_val,), 3, dtype=torch.long))
                
        # Verify targets (left-shifted, ending with 3)
        expected_targets = torch.cat([inputs[:, 1:], torch.full((num_samples, 1), 3, dtype=torch.long)], dim=1)
        assert torch.equal(targets, expected_targets)
        
        results[thread_id] = "OK"
    except Exception as e:
        results[thread_id] = str(e)


def test_concurrency_abc_task_fixed():
    """Verify thread-safety of generate_abc_task with a fixed n."""
    num_threads = 10
    num_samples = 200
    n_max = 15
    n = 10
    
    barrier = threading.Barrier(num_threads)
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            futures.append(
                executor.submit(
                    run_concurrent_abc_worker,
                    barrier,
                    num_samples,
                    n_max,
                    n,
                    i,
                    results
                )
            )
        concurrent.futures.wait(futures)
        
    for thread_id, status in results.items():
        assert status == "OK", f"Thread {thread_id} failed with error: {status}"


def test_concurrency_abc_task_variable():
    """Verify thread-safety of generate_abc_task with variable n (None)."""
    num_threads = 10
    num_samples = 200
    n_max = 15
    n = None
    
    barrier = threading.Barrier(num_threads)
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(num_threads):
            futures.append(
                executor.submit(
                    run_concurrent_abc_worker,
                    barrier,
                    num_samples,
                    n_max,
                    n,
                    i,
                    results
                )
            )
        concurrent.futures.wait(futures)
        
    for thread_id, status in results.items():
        assert status == "OK", f"Thread {thread_id} failed with error: {status}"


# -------------------------------------------------------------------------
# Large Batch Scale & Memory Profile Tests
# -------------------------------------------------------------------------

def test_large_scale_copy_task():
    """Stress test generate_copy_task at large scales (num_samples=50,000)."""
    num_samples = 50000
    length = 50
    vocab_size = 100
    
    # Enable tracemalloc to profile memory
    tracemalloc.start()
    start_time = time.perf_counter()
    
    inputs, targets = generate_copy_task(num_samples, length, vocab_size)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency = end_time - start_time
    
    # Calculate exact tensor memory usage
    tensor_memory_mb = (inputs.nelement() * inputs.element_size() + targets.nelement() * targets.element_size()) / 1024 / 1024
    
    # Print measurements for the logs/test output
    print(f"\n[Copy Task 50k] Latency: {latency:.4f}s")
    print(f"[Copy Task 50k] Python Peak Memory (tracemalloc): {peak / 1024 / 1024:.2f} MB")
    print(f"[Copy Task 50k] Tensors Size: {tensor_memory_mb:.2f} MB")
    
    # Assertions on outputs
    assert inputs.shape == (50000, 101)
    assert targets.shape == (50000, 101)
    
    # Assert latency is reasonable (should be < 1.0s on modern CPU since it's fully vectorized)
    assert latency < 1.0, f"Copy task latency {latency:.4f}s exceeded 1.0s threshold"
    
    # Verify tensor size is exact
    expected_tensor_mb = (50000 * 101 * 8 * 2) / 1024 / 1024 # 80.8 MB
    assert abs(tensor_memory_mb - expected_tensor_mb) < 1e-4


def test_large_scale_abc_task_fixed():
    """Stress test generate_abc_task (fixed n) at large scales (num_samples=50,000)."""
    num_samples = 50000
    n_max = 50
    n = 30
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    inputs, targets = generate_abc_task(num_samples, n_max, n)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency = end_time - start_time
    
    tensor_memory_mb = (inputs.nelement() * inputs.element_size() + targets.nelement() * targets.element_size()) / 1024 / 1024
    
    print(f"\n[ABC Task Fixed 50k] Latency: {latency:.4f}s")
    print(f"[ABC Task Fixed 50k] Python Peak Memory (tracemalloc): {peak / 1024 / 1024:.2f} MB")
    print(f"[ABC Task Fixed 50k] Tensors Size: {tensor_memory_mb:.2f} MB")
    
    assert inputs.shape == (50000, 90)
    assert targets.shape == (50000, 90)
    
    # Vectorized code should be very fast (< 0.2s)
    assert latency < 0.5, f"ABC task (fixed) latency {latency:.4f}s exceeded 0.5s threshold"
    
    expected_tensor_mb = (50000 * 90 * 8 * 2) / 1024 / 1024 # 72.0 MB
    assert abs(tensor_memory_mb - expected_tensor_mb) < 1e-4


def test_large_scale_abc_task_variable():
    """Stress test generate_abc_task (variable n) at large scales (num_samples=50,000)."""
    num_samples = 50000
    n_max = 50
    n = None
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    inputs, targets = generate_abc_task(num_samples, n_max, n)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency = end_time - start_time
    
    tensor_memory_mb = (inputs.nelement() * inputs.element_size() + targets.nelement() * targets.element_size()) / 1024 / 1024
    
    print(f"\n[ABC Task Variable 50k] Latency: {latency:.4f}s")
    print(f"[ABC Task Variable 50k] Python Peak Memory (tracemalloc): {peak / 1024 / 1024:.2f} MB")
    print(f"[ABC Task Variable 50k] Tensors Size: {tensor_memory_mb:.2f} MB")
    
    assert inputs.shape == (50000, 150)
    assert targets.shape == (50000, 150)
    
    # Note: Variable n task has a python loop over num_samples:
    # `for i in range(num_samples): ...`
    # Let's see how fast it runs. 50k iterations of tensor slice assignment in Python might take longer.
    # We will assert that it completes within 3.0 seconds.
    assert latency < 3.0, f"ABC task (variable) latency {latency:.4f}s exceeded 3.0s threshold"
    
    expected_tensor_mb = (50000 * 150 * 8 * 2) / 1024 / 1024 # 120.0 MB
    assert abs(tensor_memory_mb - expected_tensor_mb) < 1e-4


# -------------------------------------------------------------------------
# Boundary & Extreme Value Tests
# -------------------------------------------------------------------------

def test_boundaries_copy_task():
    """Verify copy task at minimal and larger boundary values."""
    # Minimal possible inputs
    inputs_min, targets_min = generate_copy_task(num_samples=1, length=1, vocab_size=2)
    assert inputs_min.shape == (1, 3)
    assert targets_min.shape == (1, 3)
    assert torch.equal(inputs_min, torch.tensor([[0, 1, 0]], dtype=torch.long))
    assert torch.equal(targets_min, torch.tensor([[1, 0, 0]], dtype=torch.long))
    
    # Large inputs (extreme vocab/length)
    inputs_large, targets_large = generate_copy_task(num_samples=10, length=1000, vocab_size=100000)
    assert inputs_large.shape == (10, 2001)
    assert targets_large.shape == (10, 2001)
    
    # Verify sequence structure on large vocab
    delimiter = 99999
    for seq in inputs_large:
        assert seq[1000].item() == delimiter
        assert torch.equal(seq[:1000], seq[1001:])


def test_boundaries_abc_task():
    """Verify ABC task at minimal and larger boundary values."""
    # Minimal possible inputs (fixed n)
    inputs_min, targets_min = generate_abc_task(num_samples=1, n_max=1, n=1)
    assert inputs_min.shape == (1, 3)
    assert targets_min.shape == (1, 3)
    assert torch.equal(inputs_min, torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert torch.equal(targets_min, torch.tensor([[1, 2, 3]], dtype=torch.long))
    
    # Minimal possible inputs (variable n)
    inputs_var, targets_var = generate_abc_task(num_samples=1, n_max=1, n=None)
    assert inputs_var.shape == (1, 3)
    assert targets_var.shape == (1, 3)
    assert torch.equal(inputs_var, torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert torch.equal(targets_var, torch.tensor([[1, 2, 3]], dtype=torch.long))
    
    # Large inputs
    inputs_large, targets_large = generate_abc_task(num_samples=10, n_max=1000, n=None)
    assert inputs_large.shape == (10, 3000)
    assert targets_large.shape == (10, 3000)
