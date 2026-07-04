import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class LiquidStateMachine(nn.Module):
    """
    Liquid State Machine (LSM) reservoir computing model.
    Only the readout weights are trained; the reservoir and input weights are frozen.
    """
    def __init__(self, input_size: int, reservoir_size: int, output_size: int, spectral_radius: float = 0.95, sparsity: float = 0.1):
        super().__init__()
        
        if input_size <= 0 or reservoir_size <= 0 or output_size <= 0:
            raise ValueError("Dimensions must be positive")
            
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        
        # Initialize reservoir weights W_res
        trials = 0
        while True:
            trials += 1
            if sparsity >= 1.0 or reservoir_size == 0 or trials > 100:
                W_res = torch.zeros(reservoir_size, reservoir_size)
                break
                
            W_res = torch.randn(reservoir_size, reservoir_size)
            total_elements = reservoir_size * reservoir_size
            num_zeros = int(total_elements * sparsity)
            mask = torch.ones(total_elements)
            if num_zeros > 0:
                mask[:num_zeros] = 0.0
            perm = torch.randperm(total_elements)
            mask = mask[perm].view(reservoir_size, reservoir_size)
            W_res = W_res * mask
            
            eigenvalues = np.linalg.eigvals(W_res.numpy())
            curr_radius = np.max(np.abs(eigenvalues))
            if curr_radius > 1e-5:
                W_res = W_res * (spectral_radius / curr_radius)
                break
                
        self.W_res = nn.Parameter(W_res, requires_grad=False)
        
        # Initialize input weights W_in
        # Using a standard uniform distribution scaled for reservoir input
        W_in = (torch.rand(input_size, reservoir_size) * 2.0 - 1.0) * 0.1
        self.weight_ih = nn.Parameter(W_in, requires_grad=False)
        
        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(reservoir_size), requires_grad=False)
        
        # Readout layer (trainable)
        self.readout = nn.Linear(reservoir_size, output_size)
        
    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Liquid State Machine.
        
        Args:
            x (torch.Tensor): 2D long tensor of token IDs shape (batch, seq_len) OR
                              3D float tensor of features shape (batch, seq_len, input_size).
            state (torch.Tensor, optional): Initial reservoir state of shape (batch, reservoir_size).
            
        Returns:
            logits (torch.Tensor): Readout logits of shape (batch, seq_len, output_size).
            h_t (torch.Tensor): Final reservoir state of shape (batch, reservoir_size).
        """
        if x.dim() == 2:
            # Token IDs: convert to one-hot float features
            x = nn.functional.one_hot(x, num_classes=self.input_size).float()
        elif x.dim() == 3:
            if x.shape[2] != self.input_size:
                raise ValueError(f"Input features dimension {x.shape[2]} does not match input_size {self.input_size}")
        else:
            raise ValueError(f"Unsupported input dimension {x.dim()}. Expected 2 or 3.")
            
        batch_size, seq_len, _ = x.shape
        
        if state is None:
            h_t = torch.zeros(batch_size, self.reservoir_size, device=x.device, dtype=x.dtype)
        else:
            if state.shape[0] != batch_size or state.shape[1] != self.reservoir_size:
                raise ValueError(f"State shape size mismatch: expected {(batch_size, self.reservoir_size)}, got {state.shape}")
            if state.device != x.device:
                raise ValueError(f"State device {state.device} does not match input device {x.device}")
            h_t = state.to(device=x.device, dtype=x.dtype)
            
        if seq_len == 0:
            logits = torch.zeros(batch_size, 0, self.output_size, device=x.device, dtype=x.dtype)
            return logits, h_t
            
        states = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            # Update reservoir state: h_t = tanh(x_t @ W_in + h_prev @ W_res + bias)
            h_t = torch.tanh(x_t @ self.weight_ih + h_t @ self.W_res + self.bias)
            states.append(h_t)
            
        stacked_states = torch.stack(states, dim=1)
        logits = self.readout(stacked_states)
        
        return logits, h_t
