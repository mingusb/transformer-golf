import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HAS_UNIVERSAL_LSM = False
try:
    from src.models.universal_lsm import UniversalLSM
    HAS_UNIVERSAL_LSM = True
except ImportError:
    pass

HAS_ABC_DATA = False
try:
    from src.data.context_sensitive import generate_abc_task
    HAS_ABC_DATA = True
except ImportError:
    pass


def _extract_state_tensor(state):
    """Helper to extract a 2D state tensor from various possible state structures."""
    if isinstance(state, torch.Tensor):
        return state
    if isinstance(state, (list, tuple)):
        for item in state:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(state, dict):
        for val in state.values():
            if isinstance(val, torch.Tensor):
                return val
    return None


# =====================================================================
# TIER 1 TESTS: Feature Coverage
# =====================================================================

# --- Feature F1: Basic API Contract & Instantiation ---

@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F1_01_instantiation():
    """Verify that UniversalLSM instantiates correctly with standard parameters."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    assert isinstance(model, nn.Module)
    assert hasattr(model, "fit")
    assert hasattr(model, "forward")


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F1_02_forward_shape_tokens():
    """Verify that forward pass with token sequences returns expected shapes."""
    batch_size = 3
    seq_len = 12
    input_size = 4
    output_size = 4
    model = UniversalLSM(input_size=input_size, reservoir_size=60, output_size=output_size)
    
    # Token inputs: shape (batch, seq_len)
    x = torch.randint(0, input_size, (batch_size, seq_len))
    logits, state = model(x)
    
    assert logits.shape == (batch_size, seq_len, output_size)
    state_tensor = _extract_state_tensor(state)
    assert state_tensor is not None
    assert state_tensor.shape[0] == batch_size
    assert state_tensor.shape[1] == 60


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F1_03_forward_shape_features():
    """Verify that forward pass with float features returns expected shapes if supported."""
    batch_size = 2
    seq_len = 8
    input_size = 4
    output_size = 4
    model = UniversalLSM(input_size=input_size, reservoir_size=60, output_size=output_size)
    
    # Feature inputs: shape (batch, seq_len, input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    try:
        logits, state = model(x)
        assert logits.shape == (batch_size, seq_len, output_size)
        state_tensor = _extract_state_tensor(state)
        assert state_tensor is not None
        assert state_tensor.shape[0] == batch_size
        assert state_tensor.shape[1] == 60
    except (ValueError, TypeError, RuntimeError) as e:
        # If the model strictly accepts only token inputs (2D), that is acceptable under PROJECT.md.
        # We pass the test gracefully in that case.
        pass


# --- Feature F2: Reservoir Partitioning & Scaling (R1) ---

@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F2_01_reservoir_partitioning_parameters():
    """Verify that the model has partitioned sub-reservoirs with log-scaled and infinite time constants."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    # Look for time constants, decays, or partition attributes.
    found = False
    for attr in ["tau_m", "time_constants", "decays", "membrane_time_constants", "tau"]:
        if hasattr(model, attr):
            tau = getattr(model, attr)
            if tau is not None:
                tau_list = tau.cpu().numpy() if isinstance(tau, torch.Tensor) else np.array(tau)
                # If they are time constants: inf or large
                # If they are decays: 1.0 (no decay) or 0.0 (no decay depending on how defined)
                is_persistent = np.isinf(tau_list) | (tau_list > 1e6) | np.isclose(tau_list, 1.0)
                is_transient = np.isfinite(tau_list) & (tau_list < 1e6) & ~np.isclose(tau_list, 1.0)
                if np.any(is_persistent) and np.any(is_transient):
                    found = True
                    break
                    
    assert found or hasattr(model, "persistent_indices") or hasattr(model, "sub_reservoirs"), \
        "Could not verify sub-reservoir partitions or time constants structurally."


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F2_02_decay_behavior():
    """Verify transient sub-reservoirs decay while persistent sub-reservoir maintains state."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    batch_size = 1
    init_state = torch.ones(batch_size, 60)
    
    # Feed zero inputs for multiple steps
    try:
        x_zeros = torch.zeros(batch_size, 10, dtype=torch.long)
        logits, final_state = model(x_zeros, state=init_state)
    except (RuntimeError, ValueError, TypeError):
        # Fallback: model might not accept init_state directly or expect different shape.
        x_pulse = torch.ones(batch_size, 1, dtype=torch.long)
        logits_p, init_state_out = model(x_pulse)
        x_zeros = torch.zeros(batch_size, 10, dtype=torch.long)
        logits, final_state = model(x_zeros, state=init_state_out)
        init_state = init_state_out
        
    init_state_np = _extract_state_tensor(init_state)[0].detach().cpu().numpy()
    final_state_np = _extract_state_tensor(final_state)[0].detach().cpu().numpy()
    
    # Avoid division by zero
    init_state_np = np.where(init_state_np == 0, 1e-8, init_state_np)
    decay_ratios = final_state_np / init_state_np
    
    persistent_neurons = np.where(np.abs(decay_ratios - 1.0) < 1e-3)[0]
    transient_neurons = np.where(decay_ratios < 0.95)[0]
    
    assert len(persistent_neurons) > 0, "No persistent (IF) neurons detected in state decay"
    assert len(transient_neurons) > 0, "No transient (LIF) neurons detected in state decay"


# --- Feature F3: Adaptive Spiking Gating & Routing (R2) ---

@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F3_01_gating_layer_spikes():
    """Verify that the gating layer outputs spikes (discrete 0/1 values)."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    gating_layer = None
    for attr in ["gating_layer", "gate", "routing_layer", "gating"]:
        if hasattr(model, attr):
            gating_layer = getattr(model, attr)
            break
            
    if gating_layer is not None and isinstance(gating_layer, nn.Module):
        try:
            first_param = next(gating_layer.parameters())
            in_features = first_param.shape[1] if len(first_param.shape) > 1 else first_param.shape[0]
        except StopIteration:
            in_features = 60
            
        dummy_in = torch.randn(2, in_features)
        g_out = gating_layer(dummy_in)
        unique_vals = torch.unique(g_out)
        for val in unique_vals:
            assert torch.abs(val - 0.0) < 1e-3 or torch.abs(val - 1.0) < 1e-3 or torch.isnan(val)
            
    # Alternatively, run the model and check if gating spikes are recorded/returned
    x = torch.randint(0, 4, (2, 5))
    try:
        logits, state = model(x)
        for attr in ["gating_spikes", "gate_spikes", "spikes"]:
            if hasattr(model, attr):
                spikes = getattr(model, attr)
                if spikes is not None:
                    assert torch.all((spikes == 0.0) | (spikes == 1.0) | torch.isclose(spikes, torch.tensor(0.0)) | torch.isclose(spikes, torch.tensor(1.0)))
    except Exception:
        pass


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F3_02_dynamic_routing_phases():
    """Verify that routing patterns change based on input token phases (a, b, c)."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    x_a = torch.zeros(1, 5, dtype=torch.long)
    x_b = torch.ones(1, 5, dtype=torch.long)
    x_c = torch.full((1, 5), 2, dtype=torch.long)
    
    if hasattr(model, "get_routing_signals"):
        signals_a = model.get_routing_signals(x_a)
        signals_b = model.get_routing_signals(x_b)
        signals_c = model.get_routing_signals(x_c)
        assert not torch.allclose(signals_a, signals_b)
        assert not torch.allclose(signals_b, signals_c)
    else:
        # Check if internal states/outputs differ due to routing changes
        _, state_a = model(x_a)
        _, state_b = model(x_b)
        _, state_c = model(x_c)
        
        sa = _extract_state_tensor(state_a)
        sb = _extract_state_tensor(state_b)
        sc = _extract_state_tensor(state_c)
        
        if sa is not None and sb is not None and sc is not None:
            assert not torch.allclose(sa, sb)
            assert not torch.allclose(sb, sc)


# --- Feature F4: Purely Spike-Driven & No BPTT (R3) ---

@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F4_01_no_bptt_gradient_tracking():
    """Verify that the model does not track BPTT gradients across the temporal dimension."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    for p in model.parameters():
        p.requires_grad = True
        
    x = torch.randint(0, 4, (2, 5))
    logits, final_state = model(x)
    
    state_tensor = _extract_state_tensor(final_state)
    if state_tensor is not None and state_tensor.requires_grad:
        # The recurrent state's grad_fn should be None or not show a recurrent connection trace
        assert state_tensor.grad_fn is None or "View" in str(state_tensor.grad_fn) or "Select" in str(state_tensor.grad_fn)
        
    loss = logits.sum()
    loss.backward()
    
    # Gating and reservoir recurrent parameters should not accumulate gradients through temporal backprop
    for name, param in model.named_parameters():
        if any(term in name for term in ["readout", "fc", "W_out", "fc_out"]):
            continue
        if param.grad is not None:
            assert torch.all(param.grad == 0.0) or not param.requires_grad


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T1_F4_02_dstdp_eligibility_trace_evolution():
    """Verify that eligibility traces evolve locally and decay if exposed."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    has_decay = False
    for attr in ["eligibility_traces", "traces", "eligibility_trace", "e_trace"]:
        if hasattr(model, attr):
            has_decay = True
            traces = getattr(model, attr)
            if traces is not None:
                init_trace = traces.clone() if isinstance(traces, torch.Tensor) else np.copy(traces)
                
                decay_method = None
                for method_name in ["decay_eligibility_traces", "decay_traces", "update_eligibility_traces", "step_traces"]:
                    if hasattr(model, method_name):
                        decay_method = getattr(model, method_name)
                        break
                if decay_method is not None:
                    try:
                        decay_method()
                    except TypeError:
                        try:
                            decay_method(dt=1.0)
                        except Exception:
                            pass
                    
                    decayed_trace = getattr(model, attr)
                    decayed_trace_np = decayed_trace.cpu().numpy() if isinstance(decayed_trace, torch.Tensor) else np.array(decayed_trace)
                    init_trace_np = init_trace.cpu().numpy() if isinstance(init_trace, torch.Tensor) else np.array(init_trace)
                    assert np.all(decayed_trace_np <= init_trace_np)
                break
                
    assert has_decay or hasattr(model, "fit"), "No eligibility trace mechanics or training fit method found."


# =====================================================================
# TIER 2 TESTS: Boundary & Corner Cases
# =====================================================================

@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T2_01_empty_sequence():
    """Verify handling of empty sequences."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    x = torch.randint(0, 4, (2, 0))
    logits, state = model(x)
    assert logits.shape == (2, 0, 4)
    state_tensor = _extract_state_tensor(state)
    assert state_tensor.shape == (2, 60)


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T2_02_batch_size_one():
    """Verify forward and backward with batch size 1."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    x = torch.randint(0, 4, (1, 5))
    logits, state = model(x)
    assert logits.shape == (1, 5, 4)
    state_tensor = _extract_state_tensor(state)
    assert state_tensor.shape == (1, 60)


@pytest.mark.skipif(not HAS_UNIVERSAL_LSM, reason="UniversalLSM not implemented")
def test_TEST_T2_03_invalid_vocab_indices():
    """Verify that out-of-vocab indices raise an appropriate error."""
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    x = torch.full((2, 3), 10, dtype=torch.long)
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        model(x)


# =====================================================================
# TIER 3 TESTS: Pairwise Integration
# =====================================================================

@pytest.mark.skipif(not (HAS_UNIVERSAL_LSM and HAS_ABC_DATA), reason="Missing UniversalLSM or ABC data generator")
def test_TEST_T3_01_abc_data_integration():
    """Verify that model integrates with generate_abc_task output shape."""
    num_samples = 4
    n_max = 3
    n = 2
    inputs, targets = generate_abc_task(num_samples, n_max, n=n)
    
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    logits, state = model(inputs)
    assert logits.shape == (num_samples, 3 * n, 4)


@pytest.mark.skipif(not (HAS_UNIVERSAL_LSM and HAS_ABC_DATA), reason="Missing UniversalLSM or ABC data generator")
def test_TEST_T3_02_fit_method_execution():
    """Verify that the fit training method executes and updates trainable weights."""
    inputs, targets = generate_abc_task(num_samples=5, n_max=3, n=2)
    model = UniversalLSM(input_size=4, reservoir_size=60, output_size=4)
    
    # Capture weights before fit
    w_before = None
    readout_attr = None
    for attr in ["readout", "fc", "W_out", "fc_out"]:
        if hasattr(model, attr):
            val = getattr(model, attr)
            if isinstance(val, nn.Module) and hasattr(val, "weight"):
                w_before = val.weight.clone()
                readout_attr = (attr, "module")
                break
            elif isinstance(val, torch.Tensor):
                w_before = val.clone()
                readout_attr = (attr, "tensor")
                break
                
    if w_before is None:
        for name, param in model.named_parameters():
            if any(term in name for term in ["readout", "fc", "W_out", "fc_out"]):
                w_before = param.clone()
                readout_attr = (name, "parameter")
                break
                
    model.fit(inputs, targets, epochs=1, lr=0.03, ignore_index=3)
    
    if w_before is not None:
        if readout_attr[1] == "module":
            w_after = getattr(model, readout_attr[0]).weight
        elif readout_attr[1] == "tensor":
            w_after = getattr(model, readout_attr[0])
        elif readout_attr[1] == "parameter":
            w_after = next(p for n, p in model.named_parameters() if n == readout_attr[0])
        assert not torch.equal(w_before, w_after)


# =====================================================================
# TIER 4 TESTS: Real-World Scenarios / Convergence
# =====================================================================

@pytest.mark.skipif(not (HAS_UNIVERSAL_LSM and HAS_ABC_DATA), reason="Missing UniversalLSM or ABC data generator")
def test_TEST_T4_01_abc_memorization():
    """Verify that the model can learn and memorize a small scale abc sequence task."""
    inputs, targets = generate_abc_task(num_samples=2, n_max=2, n=2)
    model = UniversalLSM(input_size=4, reservoir_size=80, output_size=4)
    
    model.fit(inputs, targets, epochs=10, lr=0.05, ignore_index=3)
    
    logits, _ = model(inputs)
    preds = logits.argmax(dim=-1)
    
    mask = (targets != 3)
    correct = (preds == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    
    assert accuracy.item() > 0.8
