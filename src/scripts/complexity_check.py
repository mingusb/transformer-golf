import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    import tests.conftest
except ImportError:
    pass

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from src.symbolic.solver import generate_routing_matrix, synthesize_circuit
from src.models.map_reduce_mlp import compile_mlp_from_circuit

def fit_degree(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    return coeffs[0]

def run_benchmarks(max_len=10, steps=5):
    print("Scaling Logs")
    lengths = np.linspace(1, max_len, steps, dtype=int)
    param_counts = []
    forward_times = []
    mem_usages = []
    
    alphabet = ['a', 'b']
    for N in lengths:
        N = int(N)
        if N <= 0:
            raise ValueError("Sequence length must be positive")
        if N > 100:
            raise ValueError("Sequence length exceeds limits")
        R = generate_routing_matrix(N, N)
        circuit = synthesize_circuit(seq_len=N, num_outputs=N, alphabet=alphabet, routing_matrix=R)
        model = compile_mlp_from_circuit(circuit, alphabet_size=2)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_counts.append(param_count)
        
        # Profile execution time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        X = torch.zeros(8, N, 2, device=device)
        # Warmup
        for _ in range(5):
            _ = model(X)
        t0 = time.perf_counter()
        for _ in range(20):
            _ = model(X)
        t1 = time.perf_counter()
        forward_time = (t1 - t0) / 20.0
        forward_times.append(forward_time)
        
        # Profile memory usage (handling CPU fallback safely)
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = model(X)
            mem_usage = float(torch.cuda.max_memory_allocated())
        else:
            # CPU fallback: estimate activation memory + parameter memory
            param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
            num_vars = len(model.inputs) + len(model.gates_modules)
            act_mem = X.shape[0] * X.shape[2] * num_vars * 4  # float32 = 4 bytes
            mem_usage = float(param_mem + act_mem)
        mem_usages.append(mem_usage)
        
        print(f"N: {N} | Parameters: {param_count} | Time: {forward_time:.6f}s | Memory: {mem_usage:.1f} bytes")
        
    if len(lengths) >= 2:
        deg_params = fit_degree(lengths, param_counts)
        deg_time = fit_degree(lengths, forward_times)
        deg_mem = fit_degree(lengths, mem_usages)
        print(f"Fit Degree (Parameters): {deg_params:.4f}")
        print(f"Fit Degree (Time): {deg_time:.4f}")
        print(f"Fit Degree (Memory): {deg_mem:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    
    if args.max_len < 0:
        raise ValueError("Sequence length cannot be negative")
        
    run_benchmarks(args.max_len, args.steps)
