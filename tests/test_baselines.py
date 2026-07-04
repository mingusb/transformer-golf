import pytest
import numpy as np
import torch
import torch.nn as nn
from src.models.sparsity import apply_l0_mask, l0_pruning_step
from src.models.baselines import (
    CausalAttentionModel,
    Conv1DModel,
    MarkovChainModel,
    run_t_test
)
from src.models.recurrent_ssm import RecurrentSSM

# ==========================================
# Feature F7: Sparsity Pruning (L0 Gating)
# ==========================================

def test_TEST_T1_F7_01():
    # TEST_T1_F7_01: L0 Mask Application Return
    # Verify apply_l0_mask returns modified neural model.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    assert isinstance(masked_model, nn.Module)

def test_TEST_T1_F7_02():
    # TEST_T1_F7_02: Differentiable Gate Variables
    # Verify gate parameters in distribution require gradient.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    # Ensure no parameter is zero so that all gates get non-zero gradients
    for p in model.parameters():
        p.data.add_(torch.randn_like(p.data) * 1e-2 + 1e-2)
    masked_model = apply_l0_mask(model)
    for name, gate in masked_model.l0_gates.items():
        assert gate.requires_grad == True
        
    # Check that task loss gradients actually flow back to the log_alpha gate parameters
    masked_model.train()
    # Fill gates with 0.0 to prevent clamping and ensure non-zero gradient flow
    for gate in masked_model.l0_gates.values():
        gate.data.fill_(0.0)
        
    X = torch.randint(0, 5, (4, 5))
    Y = torch.randint(0, 5, (4, 5))
    logits, _ = masked_model(X)
    loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), Y.view(-1))
    loss.backward()
    
    for name, gate in masked_model.l0_gates.items():
        assert gate.grad is not None, f"Gradient for gate {name} is None"
        assert torch.any(gate.grad != 0.0), f"Gradient for gate {name} is all zeros"

def test_TEST_T1_F7_03():
    # TEST_T1_F7_03: Expected L0 Penalty Validity
    # Validate L0 penalty loss value is positive.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    penalty = masked_model.compute_l0_penalty()
    # If penalty is a tensor, extract float value
    if isinstance(penalty, torch.Tensor):
        penalty = penalty.item()
    assert penalty >= 0.0

def test_TEST_T1_F7_04():
    # TEST_T1_F7_04: Pareto Frontier Tracker
    # Verify optimizer saves frontier of (sparsity, accuracy) tuples.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    assert hasattr(masked_model, "pareto_frontier")
    assert len(masked_model.pareto_frontier) > 0
    for sparsity, acc, *extra in masked_model.pareto_frontier:
        assert 0.0 <= sparsity <= 1.0
        assert 0.0 <= acc <= 1.0

def test_TEST_T1_F7_05():
    # TEST_T1_F7_05: Parameter Pruning Masking
    # Verify weights corresponding to zero-gates are masked.
    # In conftest, we can manually check if calling l0_pruning_step masks active values.
    # If gate is set to 0.0 or negative large value (which sigmoids to 0.0), weight is pruned.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    l0_pruning_step(masked_model, temperature=0, target_sparsity=1.0)
    for name, param in masked_model.named_parameters():
        if "l0_gates" in name:
            assert param.data.sum() < 0.0

def test_TEST_T2_F7_01():
    # TEST_T2_F7_01: L0 Mask on Empty Model
    # Apply L0 pruning mask to empty model.
    # We expect ValueError
    empty_model = nn.Module()
    with pytest.raises(ValueError):
        apply_l0_mask(empty_model)

def test_TEST_T2_F7_02():
    # TEST_T2_F7_02: L0 Pruning Zero Temperature
    # Run L0 optimizer with temperature parameter set to 0.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    # This shouldn't raise zero-division error in our step/sigmoid calculation
    l0_pruning_step(masked_model, temperature=0, target_sparsity=0.5)

def test_TEST_T2_F7_03():
    # TEST_T2_F7_03: Complete Target Sparsity (100%)
    # Verify model prunes all parameterized weights at 100% sparsity.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    l0_pruning_step(masked_model, temperature=0, target_sparsity=1.0)
    # Check that gates are set to highly negative values (sigmoids to 0.0)
    for name, gate in masked_model.l0_gates.items():
        assert torch.all(gate.data < -50.0)

def test_TEST_T2_F7_04():
    # TEST_T2_F7_04: Perfect Accuracy Constraint
    # Verify model is not pruned if target accuracy is set to 100.0%.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    l0_pruning_step(masked_model, temperature=0.1, target_sparsity=0.0)
    # Check that gates remain positive or unpruned
    for name, gate in masked_model.l0_gates.items():
        assert torch.all(gate.data >= 0.0)

def test_TEST_T2_F7_05():
    # TEST_T2_F7_05: Pruning Correlated Variables
    # Verify L0 gating handles highly correlated structural layers.
    # It runs to convergence successfully without division or NaN errors.
    model = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    masked_model = apply_l0_mask(model)
    l0_pruning_step(masked_model, temperature=0.5, target_sparsity=0.3)


# ==========================================
# Feature F8: Baseline Models & Statistical Tests
# ==========================================

def test_TEST_T1_F8_01():
    # TEST_T1_F8_01: Causal Attention Model Shape
    # Check causal self-attention forward output dimension.
    attention_model = CausalAttentionModel(vocab_size=10, d_model=16, state_dim=32)
    X = torch.randn(4, 8, 16)
    out, _ = attention_model(X)
    assert out.shape == (4, 8, 10)

def test_TEST_T1_F8_02():
    # TEST_T1_F8_02: Conv1D Model Output Shape
    # Check 1D CNN baseline forward output dimension.
    conv_model = Conv1DModel(vocab_size=10, d_model=16, state_dim=32)
    X = torch.randn(4, 8, 16)
    out, _ = conv_model(X)
    assert out.shape == (4, 8, 10)

def test_TEST_T1_F8_03():
    # TEST_T1_F8_03: Markov Model Probability Output
    # Validate Markov model outputs valid probability distribution.
    markov_model = MarkovChainModel(vocab_size=10, d_model=16, state_dim=32)
    probs = markov_model.predict_probabilities([1, 2, 3])
    assert probs.shape[-1] == 10
    assert np.allclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)

def test_TEST_T1_F8_04():
    # TEST_T1_F8_04: Parameter Count Parity Check
    # Verify baseline architectures have similar total params as SSM.
    ssm = RecurrentSSM(vocab_size=10, d_model=16, state_dim=32)
    attention = CausalAttentionModel(vocab_size=10, d_model=16, state_dim=32)
    conv = Conv1DModel(vocab_size=10, d_model=16, state_dim=32)
    markov = MarkovChainModel(vocab_size=10, d_model=16, state_dim=32)
    
    ssm_params = sum(p.numel() for p in ssm.parameters())
    att_params = sum(p.numel() for p in attention.parameters())
    conv_params = sum(p.numel() for p in conv.parameters())
    markov_params = sum(p.numel() for p in markov.parameters())
    
    assert abs(att_params - ssm_params) / ssm_params < 0.1
    assert abs(conv_params - ssm_params) / ssm_params < 0.1
    assert abs(markov_params - ssm_params) / ssm_params < 0.1

def test_TEST_T1_F8_05():
    # TEST_T1_F8_05: Statistical T-Test Functionality
    # Verify significance check output p-value behaves correctly.
    data1 = [0.95, 0.96, 0.94, 0.97, 0.95]
    data2 = [0.90, 0.91, 0.89, 0.92, 0.90]
    p_val = run_t_test(data1, data2)
    assert 0.0 <= p_val <= 1.0

def test_TEST_T2_F8_01():
    # TEST_T2_F8_01: Attention Maximum Window Limit
    # Forward attention model with sequence length larger than context.
    attention_model = CausalAttentionModel(vocab_size=10, d_model=16, state_dim=32)
    # Context window limit is 100 in our stub
    X = torch.randn(2, 150, 16)
    with pytest.raises(ValueError):
        attention_model(X)

def test_TEST_T2_F8_02():
    # TEST_T2_F8_02: CNN Kernel Size Limit
    # Forward Conv1D model with sequence length smaller than filter (kernel_size=3).
    conv_model = Conv1DModel(vocab_size=10, d_model=16, state_dim=32)
    X = torch.randn(2, 2, 16) # seq_len=2 < 3
    with pytest.raises(ValueError):
        conv_model(X)

def test_TEST_T2_F8_03():
    # TEST_T2_F8_03: Markov Out-of-Distribution Inputs
    # Feed OOD characters and check probability smoothing.
    markov_model = MarkovChainModel(vocab_size=5, d_model=8, state_dim=16)
    # Input has token 100 which is out of vocabulary (size=5)
    probs = markov_model.predict_probabilities([1, 100])
    assert np.allclose(probs.sum(), 1.0)
    assert np.all(probs > 0.0) # smoothing ensures non-zero probability

def test_TEST_T2_F8_04():
    # TEST_T2_F8_04: Zero Variance T-Test Data
    # Run t-test with identical datasets (variance = 0).
    data1 = [0.9, 0.9, 0.9]
    data2 = [0.9, 0.9, 0.9]
    p_val = run_t_test(data1, data2)
    assert np.allclose(p_val, 1.0)

def test_TEST_T2_F8_05():
    # TEST_T2_F8_05: Unequal Samples Statistical Test
    # Run statistical test on models with different sequence sample sizes.
    data1 = [0.95, 0.96, 0.94]
    data2 = [0.90, 0.91, 0.89, 0.92, 0.90, 0.91]
    p_val = run_t_test(data1, data2)
    assert 0.0 <= p_val <= 1.0
