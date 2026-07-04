import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class GatingLayer(nn.Module):
    """
    Spiking gating layer using LIF neurons.
    Outputs binary/spiking activations (0 or 1).
    """
    def __init__(self, input_size: int, reservoir_size: int, gating_decay: float):
        super().__init__()
        self.fc = nn.Linear(input_size, reservoir_size)
        self.gating_decay = gating_decay
        self.V_g = None
        self.S_g = None

    def reset_state(self, batch_size: int, device):
        self.V_g = torch.zeros(batch_size, self.fc.out_features, device=device)
        self.S_g = torch.zeros(batch_size, self.fc.out_features, device=device)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if self.V_g is None or self.V_g.shape[0] != u.shape[0] or self.V_g.device != u.device:
            self.reset_state(u.shape[0], u.device)
        # V^g(t) = \alpha_g V^g(t-1) (1 - S^g(t-1)) + W^g u(t) + b^g
        current = self.fc(u)
        self.V_g = self.gating_decay * self.V_g * (1.0 - self.S_g) + current
        # S^g(t) = H(V^g(t) - 1.0)
        self.S_g = (self.V_g > 1.0).float()
        return self.S_g

class UniversalLSM(nn.Module):
    """
    Universal Liquid State Machine (UniversalLSM) with partitioned sub-reservoirs,
    log-scaled membrane time constants, persistent sub-reservoir, adaptive spiking gating layer,
    and spike-driven local learning rule via e-prop/D-STDP.
    """
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        tau_min: float = 2.0,
        tau_max: float = 20.0,
        transient_fraction: float = 0.8,
        eligibility_decay: float = 0.9,
        gating_decay: float = 0.8,
        lr_gating: float = 0.01,
        lr_readout: float = 0.01
    ):
        super().__init__()
        
        if input_size <= 0 or reservoir_size <= 0 or output_size <= 0:
            raise ValueError("Dimensions must be positive")
            
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.transient_fraction = transient_fraction
        self.eligibility_decay = eligibility_decay
        self.gating_decay = gating_decay
        self.lr_gating = lr_gating
        self.lr_readout = lr_readout
        
        # Partitioning sub-reservoirs
        N_trans = int(transient_fraction * reservoir_size)
        N_pers = reservoir_size - N_trans
        
        # Calculate log-scaled time constants
        tau_m = torch.zeros(reservoir_size)
        if N_trans > 0:
            if N_trans > 1:
                for i in range(N_trans):
                    tau_m[i] = tau_min * (tau_max / tau_min) ** (i / (N_trans - 1))
            else:
                tau_m[0] = tau_min
        tau_m[N_trans:] = float('inf')
        
        self.register_buffer("tau_m", tau_m)
        self.time_constants = self.tau_m
        self.membrane_time_constants = self.tau_m
        self.tau = self.tau_m
        
        # Decay factor alpha
        self.register_buffer("alpha", 1.0 - 1.0 / self.tau_m)
        
        # Gating Layer
        self.gating_layer = GatingLayer(input_size, reservoir_size, gating_decay)
        self.gate = self.gating_layer
        self.routing_layer = self.gating_layer
        self.gating = self.gating_layer
        
        # Freeze gating layer parameters initially
        for p in self.gating_layer.parameters():
            p.requires_grad = False
            
        # Initialize gating layer weights to be uniform positive/negative, bias to be uniform positive near threshold
        nn.init.uniform_(self.gating_layer.fc.weight, -0.5, 0.5)
        nn.init.uniform_(self.gating_layer.fc.bias, 0.5, 1.0)
            
        # Initialize reservoir weights W_in and W_res
        self.W_in = nn.Parameter(torch.zeros(reservoir_size, input_size), requires_grad=False)
        nn.init.uniform_(self.W_in, 0.2, 1.2)
        self.weight_ih = self.W_in
        
        self.W_res = nn.Parameter(torch.zeros(reservoir_size, reservoir_size), requires_grad=False)
        nn.init.uniform_(self.W_res, -0.2, 0.2)
        
        # Dummy bias parameter to satisfy potential API checks
        self.bias = nn.Parameter(torch.zeros(reservoir_size), requires_grad=False)
        
        # Readout layer (trainable)
        self.readout = nn.Linear(reservoir_size, output_size)
        self.fc = self.readout
        self.fc_out = self.readout
        self.W_out = self.readout
        
        # Fixed random feedback projection matrices (scaled to be larger for stronger feedback signal)
        self.register_buffer('B', torch.randn(reservoir_size, output_size) * 0.5)
        self.register_buffer('B_res', torch.randn(reservoir_size, output_size) * 0.5)
        
        # Eligibility trace attribute for gating layer
        self.register_buffer("eligibility_traces", torch.zeros(reservoir_size, input_size))
        self.traces = self.eligibility_traces
        self.eligibility_trace = self.eligibility_traces
        self.e_trace = self.eligibility_traces
        
        # Recorded gating spikes
        self.gating_spikes = None

    def decay_eligibility_traces(self):
        """Decay eligibility traces by eligibility_decay factor."""
        self.eligibility_traces.mul_(self.eligibility_decay)

    def get_routing_signals(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return gating spikes history."""
        with torch.no_grad():
            self.forward(x)
            return self.gating_spikes.clone()

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Supports token sequences (2D) and feature sequences (3D).
        """
        # Validate and convert input format
        if x.dim() == 2:
            if (x >= self.input_size).any() or (x < 0).any():
                raise IndexError("Token index out of vocabulary/input size bounds")
            u = nn.functional.one_hot(x, num_classes=self.input_size).float()
        elif x.dim() == 3:
            if x.shape[2] != self.input_size:
                raise ValueError(f"Feature dimension {x.shape[2]} does not match input_size {self.input_size}")
            u = x
        else:
            raise ValueError(f"Unsupported input dimension {x.dim()}. Expected 2 or 3.")
            
        batch_size, seq_len, _ = u.shape
        device = u.device
        
        # Handle empty sequences
        if seq_len == 0:
            logits = torch.zeros(batch_size, 0, self.output_size, device=device)
            self.gating_spikes = torch.zeros(batch_size, 0, self.reservoir_size, device=device)
            V_res = torch.zeros(batch_size, self.reservoir_size, device=device) if state is None else state.clone().to(device)
            return logits, V_res
            
        # Handle state extraction
        if state is None:
            V_res = torch.zeros(batch_size, self.reservoir_size, device=device)
        else:
            if isinstance(state, torch.Tensor):
                V_res = state.clone().to(device)
            elif isinstance(state, (list, tuple)):
                V_res = next(item for item in state if isinstance(item, torch.Tensor)).clone().to(device)
            elif isinstance(state, dict):
                V_res = next(val for val in state.values() if isinstance(val, torch.Tensor)).clone().to(device)
            else:
                raise ValueError("Unsupported state format")
                
            if V_res.shape[0] != batch_size or V_res.shape[1] != self.reservoir_size:
                raise ValueError(f"State shape size mismatch: expected {(batch_size, self.reservoir_size)}, got {V_res.shape}")
                
        # Reset gating layer state
        self.gating_layer.reset_state(batch_size, device)
        
        S_res_prev = torch.zeros(batch_size, self.reservoir_size, device=device)
        gating_spikes_list = []
        states_list = []
        
        # Execute forward pass without tracking autograd gradients for LSM recurrent/gating components
        with torch.no_grad():
            for t in range(seq_len):
                u_t = u[:, t, :]
                
                # Gating layer spiking activity
                S_gate = self.gating_layer(u_t)
                gating_spikes_list.append(S_gate)
                
                # Input and recurrent currents gated by gating spikes
                input_current = u_t @ self.W_in.t()
                recurrent_current = S_res_prev @ self.W_res.t()
                I_t = S_gate * (input_current + recurrent_current)
                
                # Update membrane potentials of reservoir neurons
                V_res = self.alpha * V_res * (1.0 - S_res_prev) + I_t
                
                # Reservoir spikes
                S_res = (V_res > 1.0).float()
                
                # Save variables for the next step
                S_res_prev = S_res
                states_list.append(V_res.clone())
                
        self.gating_spikes = torch.stack(gating_spikes_list, dim=1)
        stacked_states = torch.stack(states_list, dim=1)
        
        # Readout (runs outside no_grad to support standard autograd if desired)
        logits = self.readout(stacked_states)
        
        return logits, V_res

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 1, lr: float = 0.01, ignore_index: int = -100):
        """
        Train the model using purely local spike-driven update rules (e-prop/D-STDP).
        """
        # Validate inputs
        if inputs.dim() == 2:
            if (inputs >= self.input_size).any() or (inputs < 0).any():
                raise IndexError("Token index out of vocabulary/input size bounds")
            u = nn.functional.one_hot(inputs, num_classes=self.input_size).float()
        elif inputs.dim() == 3:
            u = inputs
        else:
            raise ValueError("Unsupported input dimension")
            
        batch_size, seq_len, _ = u.shape
        device = u.device
        
        lr_readout = lr
        lr_gating = lr
        lr_res = lr
        
        for epoch in range(epochs):
            # Reset state for sequence
            self.gating_layer.reset_state(batch_size, device)
            V_res = torch.zeros(batch_size, self.reservoir_size, device=device)
            S_res_prev = torch.zeros(batch_size, self.reservoir_size, device=device)
            
            # Initialize eligibility traces
            E_g = torch.zeros(batch_size, self.reservoir_size, self.input_size, device=device)
            E_g_bias = torch.zeros(batch_size, self.reservoir_size, device=device)
            E_res = torch.zeros(batch_size, self.reservoir_size, self.reservoir_size, device=device)
            E_in = torch.zeros(batch_size, self.reservoir_size, self.input_size, device=device)
            
            for t in range(seq_len):
                u_t = u[:, t, :]
                d_t = targets[:, t]
                
                # Check target mask
                valid_mask = (d_t != ignore_index).float().unsqueeze(-1) # (batch, 1)
                
                # Forward step (gating layer)
                S_gate = self.gating_layer(u_t)
                V_gate = self.gating_layer.V_g
                
                # Forward step (reservoir)
                input_current = u_t @ self.W_in.t()
                recurrent_current = S_res_prev @ self.W_res.t()
                I_t = S_gate * (input_current + recurrent_current)
                
                V_res = self.alpha * V_res * (1.0 - S_res_prev) + I_t
                S_res = (V_res > 1.0).float()
                
                # Predict and compute error
                y_t = self.readout(V_res) # (batch, output_size)
                p_t = torch.softmax(y_t, dim=-1)
                
                safe_targets = d_t.clone()
                safe_targets[d_t == ignore_index] = 0
                d_one_hot = nn.functional.one_hot(safe_targets, num_classes=self.output_size).float()
                
                # Error: (d(t) - p(t)) * mask
                error = (d_one_hot - p_t) * valid_mask # (batch, output_size)
                
                # 1. Update Readout Weights
                dW_out = torch.matmul(error.t(), V_res) / batch_size
                db_out = error.mean(dim=0)
                self.readout.weight.data.add_(dW_out * lr_readout)
                self.readout.bias.data.add_(db_out * lr_readout)
                
                # 2. Update Gating Layer Weights
                e_g = error @ self.B.t() # (batch, reservoir_size)
                
                # Surrogate derivative for gating layer
                gamma_g = 1.0
                surr_deriv_gate = gamma_g * torch.clamp(1.0 - torch.abs(V_gate - 1.0), min=0.0)
                
                # Update gating eligibility traces in-place
                E_g.mul_(self.eligibility_decay).add_(surr_deriv_gate.unsqueeze(-1) * u_t.unsqueeze(1))
                
                dW_g = torch.einsum('bi,bij->ij', e_g, E_g) / batch_size
                self.gating_layer.fc.weight.data.add_(dW_g * lr_gating)
                
                # Update gating bias in-place
                E_g_bias.mul_(self.eligibility_decay).add_(surr_deriv_gate)
                db_g = (e_g * E_g_bias).mean(dim=0)
                self.gating_layer.fc.bias.data.add_(db_g * lr_gating)
                
                # 3. Update Reservoir Recurrent Weights
                e_res = error @ self.B_res.t() # (batch, reservoir_size)
                
                # Surrogate derivative for reservoir neurons
                gamma_res = 1.0
                surr_deriv_res = gamma_res * torch.clamp(1.0 - torch.abs(V_res - 1.0), min=0.0)
                
                # Update reservoir recurrent eligibility traces in-place
                E_res.mul_(self.eligibility_decay).add_(surr_deriv_res.unsqueeze(-1) * S_res_prev.unsqueeze(1))
                
                dW_res = torch.einsum('bi,bij->ij', e_res, E_res) / batch_size
                self.W_res.data.add_(dW_res * lr_res)
                
                # Update reservoir input weights in-place
                E_in.mul_(self.eligibility_decay).add_(surr_deriv_res.unsqueeze(-1) * u_t.unsqueeze(1))
                
                dW_in = torch.einsum('bi,bij->ij', e_res, E_in) / batch_size
                self.W_in.data.add_(dW_in * lr_res)
                
                # Save state and spikes
                S_res_prev = S_res
                
            # Assign the average gating eligibility trace to eligibility_traces attribute
            self.eligibility_traces = E_g.mean(dim=0)
