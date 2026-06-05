import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
import os
import gc

# Ensure the project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.recurrent_ssm import RecurrentSSM, make_hippo

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def test_numerical_stability():
    print("=== TASK 1: Numerical Stability of Cayley Matrix A ===")
    set_seed()
    
    vocab_size = 10
    d_model = 16
    state_dim = 64
    seq_len = 5000
    batch_size = 8
    
    # 1. Compare hippo_init vs random init eigenvalues
    model_hippo = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=True)
    model_rand = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=False)
    
    eig_hippo = torch.linalg.eigvals(model_hippo.A).abs().max().item()
    eig_rand = torch.linalg.eigvals(model_rand.A).abs().max().item()
    
    print(f"Max eigenvalue magnitude (HiPPO Init A): {eig_hippo:.6f}")
    print(f"Max eigenvalue magnitude (Random Init A): {eig_rand:.6f}")
    
    # 2. Monitor state magnitudes over 5000 steps
    # We generate random token sequences
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    for name, model in [("HiPPO Init", model_hippo), ("Random Init", model_rand)]:
        print(f"\nRunning forward pass with {name} for {seq_len} steps...")
        
        # We will manually step through the model to monitor the hidden state at each step
        x_emb = model._prepare_input(x)
        current_state = torch.zeros(batch_size, state_dim, device=x.device)
        
        state_maxs = []
        state_means = []
        state_stds = []
        
        for t in range(seq_len):
            xt = x_emb[:, t, :]
            gate = torch.sigmoid(xt @ model.gate_w.T + model.gate_b)
            proposed_state = torch.tanh(current_state @ model.A.T + xt @ model.B.T)
            next_state = (1.0 - gate) * current_state + gate * proposed_state
            
            current_state = next_state
            
            # Record stats
            abs_state = torch.abs(current_state)
            state_maxs.append(abs_state.max().item())
            state_means.append(current_state.mean().item())
            state_stds.append(current_state.std().item())
            
        print(f"Final state max abs value: {state_maxs[-1]:.6f} (Expected <= 1.0 due to sigmoid/tanh bound)")
        print(f"Max state abs value over ALL steps: {max(state_maxs):.6f}")
        print(f"State mean at step 5000: {state_means[-1]:.6f}")
        print(f"State std at step 5000: {state_stds[-1]:.6f}")
        
        # Check for NaN / Inf
        has_nan = np.isnan(state_maxs).any()
        has_inf = np.isinf(state_maxs).any()
        print(f"Has NaNs: {has_nan}, Has Infs: {has_inf}")
        
        assert max(state_maxs) <= 1.0, f"Error: State magnitude exceeded 1.0! Max was {max(state_maxs)}"
        assert not has_nan, "Error: NaN encountered in states"
        assert not has_inf, "Error: Inf encountered in states"
        
    print("Numerical stability verification: PASSED.\n")
    return eig_hippo, eig_rand

def test_gradient_stability():
    print("=== TASK 2: TBPTT vs BPTT Gradient Stability ===")
    set_seed()
    
    vocab_size = 10
    d_model = 16
    state_dim = 32
    seq_len = 1000 # Using 1000 to keep standard BPTT memory/time overhead reasonable
    batch_size = 4
    epochs = 10
    
    # Generate random training data
    X_train = torch.randint(0, vocab_size, (batch_size, seq_len))
    Y_train = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Run Standard BPTT
    print("Running standard BPTT...")
    model_bptt = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=True)
    optimizer_bptt = torch.optim.Adam(model_bptt.parameters(), lr=0.03)
    
    bptt_norms = []
    bptt_losses = []
    
    try:
        for epoch in range(epochs):
            optimizer_bptt.zero_grad()
            logits, _ = model_bptt(X_train)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y_train.reshape(-1))
            loss.backward()
            
            # Compute total gradient norm
            total_norm = 0.0
            for p in model_bptt.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            bptt_norms.append(total_norm)
            bptt_losses.append(loss.item())
            
            optimizer_bptt.step()
            print(f"  BPTT Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Grad Norm: {total_norm:.6f}")
    except Exception as e:
        print(f"  BPTT failed/errored: {e}")
        bptt_norms = [float('nan')] * epochs
        
    # Run TBPTT
    print("\nRunning TBPTT (chunk_len=5)...")
    model_tbptt = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=True)
    optimizer_tbptt = torch.optim.Adam(model_tbptt.parameters(), lr=0.03)
    
    tbptt_norms = []
    tbptt_losses = []
    
    for epoch in range(epochs):
        state = None
        epoch_norms = []
        epoch_losses = []
        chunk_len = 5
        
        for t in range(0, seq_len, chunk_len):
            optimizer_tbptt.zero_grad()
            X_chunk = X_train[:, t : t + chunk_len]
            Y_chunk = Y_train[:, t : t + chunk_len]
            
            if state is not None:
                state = state.detach()
                
            logits, next_state = model_tbptt(X_chunk, state)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y_chunk.reshape(-1))
            loss.backward()
            
            # Compute total gradient norm
            total_norm = 0.0
            for p in model_tbptt.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            epoch_norms.append(total_norm)
            epoch_losses.append(loss.item())
            
            optimizer_tbptt.step()
            state = next_state
            
        mean_norm = np.mean(epoch_norms)
        mean_loss = np.mean(epoch_losses)
        tbptt_norms.append(mean_norm)
        tbptt_losses.append(mean_loss)
        print(f"  TBPTT Epoch {epoch+1}/{epochs} | Avg Loss: {mean_loss:.4f} | Avg Grad Norm: {mean_norm:.6f}")
        
    print("\nGradient stability verification: Comparing trends...")
    print(f"Standard BPTT Grad Norms: min={min(bptt_norms):.6f}, max={max(bptt_norms):.6f}, last={bptt_norms[-1]:.6f}")
    print(f"TBPTT Grad Norms: min={min(tbptt_norms):.6f}, max={max(tbptt_norms):.6f}, last={tbptt_norms[-1]:.6f}")
    
    # A common sign of exploding gradients in standard BPTT is huge values,
    # and vanishing is extremely small values.
    # TBPTT restricts the backpropagation depth, which should keep it very stable.
    # Let's verify that TBPTT gradients neither explode (e.g. > 100.0) nor vanish to 0.0.
    assert max(tbptt_norms) < 50.0, f"Error: TBPTT gradient exploded. Max norm: {max(tbptt_norms)}"
    assert min(tbptt_norms) > 1e-5, f"Error: TBPTT gradient vanished. Min norm: {min(tbptt_norms)}"
    print("Gradient stability verification: PASSED.\n")

def test_o1_complexity(eig_hippo, eig_rand):
    print("=== TASK 3: O(1) Step Update Complexity ===")
    set_seed()
    
    vocab_size = 10
    d_model = 32
    state_dim = 128
    batch_size = 16
    
    model = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=True)
    model.eval()
    
    # We want to measure:
    # 1. Total execution time for different sequence lengths (to verify linear scaling of total forward pass)
    # 2. Timing of individual steps t in a long sequence of 5000 steps (to verify O(1) per-step time)
    
    seq_lengths = [500, 1000, 2000, 3000, 4000, 5000]
    total_times = []
    
    # Warmup
    x_warmup = torch.randint(0, vocab_size, (batch_size, 100))
    with torch.no_grad():
        for _ in range(20):
            _ = model(x_warmup)
            
    print("Measuring total execution time vs sequence length (using minimum of 10 runs to avoid transient OS noise)...")
    for L in seq_lengths:
        x_L = torch.randint(0, vocab_size, (batch_size, L))
        
        runs = 10
        times = []
        for _ in range(runs):
            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            with torch.no_grad():
                _, _ = model(x_L)
            t1 = time.perf_counter()
            gc.enable()
            times.append(t1 - t0)
            
        elapsed = min(times) # Use minimum run time
        total_times.append(elapsed)
        per_step_time = (elapsed / L) * 1e6 # in microseconds
        print(f"  Seq Len: {L:4d} | Total Time: {elapsed:.6f}s | Per-Step Time: {per_step_time:.3f} us")
        
    # Check linear fit of total time
    # Total time = a * L + b
    # R^2 should be close to 1.0 (indicating perfect linear scaling, i.e., constant per-step time)
    coeffs = np.polyfit(seq_lengths, total_times, 1)
    slope, intercept = coeffs
    
    # Compute R^2
    y_pred = np.polyval(coeffs, seq_lengths)
    y_mean = np.mean(total_times)
    ss_tot = np.sum((total_times - y_mean) ** 2)
    ss_res = np.sum((total_times - y_pred) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)
    
    print(f"Linear regression fit: Total Time = {slope*1e6:.4f} us * L + {intercept*1e3:.4f} ms")
    print(f"R^2 coefficient of determination: {r_squared:.6f}")
    
    # Now, measure the time of individual steps in a 5000-step loop
    print("\nMeasuring execution time of individual steps over 5000 steps...")
    x_5000 = torch.randint(0, vocab_size, (batch_size, 5000))
    x_emb = model._prepare_input(x_5000)
    current_state = torch.zeros(batch_size, state_dim)
    
    step_times = []
    
    gc.collect()
    gc.disable()
    with torch.no_grad():
        for t in range(5000):
            xt = x_emb[:, t, :]
            
            t0 = time.perf_counter()
            # The core recurrent step:
            gate = torch.sigmoid(xt @ model.gate_w.T + model.gate_b)
            proposed_state = torch.tanh(current_state @ model.A.T + xt @ model.B.T)
            next_state = (1.0 - gate) * current_state + gate * proposed_state
            current_state = next_state
            t1 = time.perf_counter()
            
            step_times.append(t1 - t0)
    gc.enable()
            
    # Compute averages in bins to see if there is any upward trend
    bin_size = 1000
    print("Per-step execution times averaged over 1000-step bins (in microseconds):")
    bin_means = []
    for i in range(0, 5000, bin_size):
        bin_data = step_times[i : i + bin_size]
        mean_us = np.mean(bin_data) * 1e6
        std_us = np.std(bin_data) * 1e6
        bin_means.append(mean_us)
        print(f"  Steps {i+1:4d} to {i+bin_size:4d}: Mean = {mean_us:.3f} us, Std = {std_us:.3f} us")
        
    # Check if there is a significant upward trend
    # We fit a line to the step times. If step complexity is O(1), the slope should be extremely close to 0.
    step_indices = np.arange(1, 5001)
    step_slope, _ = np.polyfit(step_indices, step_times, 1)
    # The slope is time change per step. Let's convert to microseconds per 1000 steps.
    slope_us_per_k = step_slope * 1000 * 1e6
    print(f"Trend slope: {slope_us_per_k:.6f} microseconds increase per 1000 steps")
    
    # Assertions
    assert r_squared > 0.95 or abs(slope_us_per_k) < 5.0, f"Error: Total time is not highly linear with sequence length (R^2 = {r_squared:.4f}) and step trend slope is non-negligible ({slope_us_per_k:.3f} us/k steps)"
    
    print("O(1) Step Update Complexity verification: PASSED.\n")
    
    # Save the output to a JSON structure for easy reporting
    results = {
        "eigenvalues": {
            "hippo": eig_hippo,
            "random": eig_rand
        },
        "complexity": {
            "seq_lengths": seq_lengths,
            "total_times": total_times,
            "r_squared": r_squared,
            "slope_us_per_step": slope * 1e6,
            "intercept_ms": intercept * 1e3,
            "bin_means_us": bin_means,
            "step_trend_us_per_k_steps": slope_us_per_k
        }
    }
    return results

if __name__ == "__main__":
    print("=== STARTING RECURRENT SSM STRESS TEST AND PROFILING ===\n")
    try:
        eig_hippo, eig_rand = test_numerical_stability()
        test_gradient_stability()
        results = test_o1_complexity(eig_hippo, eig_rand)
        
        # Write results to temporary JSON in results/
        os.makedirs("results", exist_ok=True)
        with open("results/recurrent_ssm_stress_results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("All tests completed successfully. Results saved to results/recurrent_ssm_stress_results.json")
    except Exception as e:
        print(f"Test suite failed with exception: {e}")
        sys.exit(1)
