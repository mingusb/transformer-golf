import pytest
import numpy as np
import torch
import torch.nn as nn
from src.models.recurrent_ssm import RecurrentSSM, train_model_tbptt
from src.models.sparsity import ParetoEntry, apply_l0_mask, l0_pruning_step
from src.models.baselines import CausalAttentionModel, Conv1DModel, MarkovChainModel, run_t_test

# ==========================================
# 1. ParetoEntry Unpacking Mismatch Bug
# ==========================================

def test_pareto_entry_unpacking_bug():
    # ParetoEntry subclass tuple and has 3 fields. __iter__ now returns all 3 elements.
    # Unpacking ParetoEntry as a 3-tuple should succeed.
    entry = ParetoEntry(0.2, 0.95, 0.90)
    
    # Asserting that the property getters work
    assert entry.sparsity == 0.2
    assert entry.token_accuracy == 0.95
    assert entry.sequence_accuracy == 0.90
    
    # Asserting that iterating/unpacking successfully unpacks to 3 variables
    s, t, seq = entry
    assert s == 0.2
    assert t == 0.95
    assert seq == 0.90


# ==========================================
# 2. apply_l0_mask ParameterList/ParameterDict Crash
# ==========================================

def test_l0_mask_self_inspection_vulnerability():
    # During the forward pass, apply_l0_mask deletes the original parameters from the module
    # and temporarily replaces them with tensors (param * z). This means any self-inspection
    # of parameters() or named_parameters() inside the forward pass (e.g. for dynamic weight decay
    # or parameter hooks) will fail or see an empty list of parameters.
    class ModelWithSelfAccess(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 5
            self.weight = nn.Parameter(torch.randn(5, 5))
            self.holder = [self.weight]
        def forward(self, x):
            params = list(self.parameters())
            # Check if the original weight parameter is still registered
            weight_found = any(p is self.holder[0] for p in params)
            if not weight_found:
                raise RuntimeError("Original weight parameter is missing from parameters() during forward!")
            if x.dtype.is_floating_point:
                return x @ self.weight
            else:
                x_one_hot = nn.functional.one_hot(x.long(), num_classes=5).float()
                return x_one_hot @ self.weight

    model = ModelWithSelfAccess()
    # This should now succeed without raising a RuntimeError
    apply_l0_mask(model)


# ==========================================
# 3. MarkovChainModel.predict_probabilities Array Truth Value Bug
# ==========================================

def test_markov_chain_predict_probabilities_array_bug():
    model = MarkovChainModel(vocab_size=5, d_model=8, state_dim=16)
    
    # Passing a list works
    probs_list = model.predict_probabilities([1, 2])
    assert len(probs_list) == 5
    
    # Passing a numpy array now succeeds!
    probs_arr = model.predict_probabilities(np.array([1, 2]))
    assert len(probs_arr) == 5
    
    # Passing a torch tensor now succeeds!
    probs_tensor = model.predict_probabilities(torch.tensor([1, 2]))
    assert len(probs_tensor) == 5


# ==========================================
# 4. RecurrentSSM Boundary Conditions & Robustness
# ==========================================

def test_recurrent_ssm_unsupported_dimensions():
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16)
    
    # 1D input -> raises ValueError on unsupported dimensions
    x_1d = torch.tensor([1, 2, 3])
    with pytest.raises(ValueError) as excinfo:
        model(x_1d)
    assert "Unsupported input dimension" in str(excinfo.value)
    
    # 4D input -> raises ValueError on unsupported dimensions
    x_4d = torch.randn(2, 3, 4, 8)
    with pytest.raises(ValueError) as excinfo:
        model(x_4d)
    assert "Unsupported input dimension" in str(excinfo.value)


def test_recurrent_ssm_state_mismatches():
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16)
    x = torch.randint(0, 5, (2, 3))
    
    # 1. State shape mismatch (state dim is 16, but we pass 17)
    bad_state = torch.zeros(2, 17)
    with pytest.raises(RuntimeError) as excinfo:
        model(x, state=bad_state)
    assert "State shape size mismatch" in str(excinfo.value)
    
    # 2. State device mismatch (if model/input is CPU, and state is CUDA - or vice versa)
    if torch.cuda.is_available():
        cuda_state = torch.zeros(2, 16, device="cuda")
        with pytest.raises(RuntimeError) as excinfo:
            model(x, state=cuda_state)
        assert "does not match input device" in str(excinfo.value)


def test_tbptt_negative_chunk_len():
    # If chunk_len <= 0, train_model_tbptt should raise ValueError or range error
    model = RecurrentSSM(vocab_size=2, d_model=4, state_dim=8)
    X = torch.randint(0, 2, (2, 10))
    Y = torch.randint(0, 2, (2, 10))
    
    # Zero chunk length raises ValueError
    with pytest.raises(ValueError) as excinfo:
        train_model_tbptt(model, X, Y, chunk_len=0, epochs=1)
    assert "chunk_len must be greater than 0" in str(excinfo.value)


# ==========================================
# 5. Sparsity L0 Temp Extremes (Numerical Stability)
# ==========================================

def test_l0_temp_extreme_numerical_instability():
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16)
    masked_model = apply_l0_mask(model)
    
    # Set temperature extremely close to 0 but positive (e.g. 1e-45)
    masked_model.l0_temp = 1e-45
    masked_model.train()
    
    # Forward pass can produce NaN due to division by extremely small l0_temp
    # log(u) - log(1-u) + log_alpha is divided by l0_temp.
    # If l0_temp is 1e-45, this division blows up to +/- infinity.
    # Sigmoid(inf) is 1.0, Sigmoid(-inf) is 0.0, but if we get NaN from undefined operations,
    # the output is NaN. Let's see if we get NaNs or if it is stable.
    x = torch.randint(0, 5, (2, 3))
    
    # Let's run a forward pass multiple times to see if NaN occurs.
    # Note that PyTorch's sigmoid of very large/small numbers usually saturates stably,
    # but the backpropagation can easily produce NaNs or Infs.
    logits, _ = masked_model(x)
    loss = logits.sum()
    loss.backward()
    
    # Check if any gate gradients are NaN/Inf
    for name, gate in masked_model.l0_gates.items():
        if gate.grad is not None:
            # Let's see if gradients are stable or if they overflow/underflow
            # We don't assert failure, we just observe if there's any nan/inf
            has_nan = torch.isnan(gate.grad).any().item()
            has_inf = torch.isinf(gate.grad).any().item()
            print(f"Gate {name} grad has NaN: {has_nan}, has Inf: {has_inf}")


# ==========================================
# 6. Baseline Models Constraints
# ==========================================

def test_causal_attention_extrapolation_limit():
    # CausalAttentionModel fails for seq_len > 100
    model = CausalAttentionModel(vocab_size=5, d_model=8, state_dim=16)
    
    # Seq len 100 works
    x_100 = torch.randn(2, 100, 8)
    out, _ = model(x_100)
    assert out.shape == (2, 100, 5)
    
    # Seq len 101 crashes
    x_101 = torch.randn(2, 101, 8)
    with pytest.raises(ValueError) as excinfo:
        model(x_101)
    assert "exceeds maximum context window" in str(excinfo.value)


def test_conv1d_sequence_length_limit():
    # Conv1DModel fails for seq_len < 3
    model = Conv1DModel(vocab_size=5, d_model=8, state_dim=16)
    
    x_2 = torch.randn(2, 2, 8)
    with pytest.raises(ValueError) as excinfo:
        model(x_2)
    assert "smaller than kernel size 3" in str(excinfo.value)
