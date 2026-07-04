import torch
import torch.nn as nn
from typing import Tuple, Optional

class StackRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, stack_width: int, stack_depth: int):
        super().__init__()
        if vocab_size <= 0 or hidden_size <= 0 or stack_width <= 0 or stack_depth <= 0:
            raise ValueError("Dimensions must be positive")
            
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Controller recurrent cell: takes concatenated embedding + stack top as input
        self.rnn_cell = nn.GRUCell(hidden_size + stack_width, hidden_size)
        
        # Project hidden state to gates (push, pop, noop) and push value v_t
        self.stack_proj = nn.Linear(hidden_size, 3 + stack_width)
        
        # Readout layer projecting hidden state to vocabulary logits
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Stack-Augmented RNN.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len)
            state (torch.Tensor, optional): Initial controller hidden state of shape (batch_size, hidden_size)
            
        Returns:
            logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size)
            stack_states (torch.Tensor): Soft stack states of shape (batch_size, seq_len, stack_depth, stack_width)
        """
        batch_size, seq_len = x.shape
        
        if (x >= self.vocab_size).any() or (x < 0).any():
            raise IndexError("Token index out of vocabulary bounds")
            
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=self.embedding.weight.dtype)
        else:
            if len(state.shape) != 2 or state.shape[0] != batch_size or state.shape[1] != self.hidden_size:
                raise RuntimeError(f"State shape size mismatch: expected {(batch_size, self.hidden_size)}, got {state.shape}")
            if state.device != x.device:
                raise RuntimeError(f"State device {state.device} does not match input device {x.device}")
            h = state.to(device=x.device, dtype=self.embedding.weight.dtype)
            
        # Initialize stack representation: S of shape (batch_size, stack_depth, stack_width)
        S = torch.zeros(batch_size, self.stack_depth, self.stack_width, device=x.device, dtype=self.embedding.weight.dtype)
        
        logits_list = []
        stack_states_list = []
        
        # Embed inputs
        embedded_x = self.embedding(x)  # shape: (batch_size, seq_len, hidden_size)
        
        for t in range(seq_len):
            x_t = embedded_x[:, t, :]  # shape: (batch_size, hidden_size)
            S_prev = S  # shape: (batch_size, stack_depth, stack_width)
            
            # Stack top S_{t-1, 0}
            stack_top = S_prev[:, 0, :]  # shape: (batch_size, stack_width)
            cell_input = torch.cat([x_t, stack_top], dim=-1)  # shape: (batch_size, hidden_size + stack_width)
            
            h = self.rnn_cell(cell_input, h)  # shape: (batch_size, hidden_size)
            
            # Project to stack operations: gates & value v_t
            proj = self.stack_proj(h)  # shape: (batch_size, 3 + stack_width)
            gate_logits = proj[:, :3]
            v_t = proj[:, 3:]
            
            # Compute push gate p_t, pop gate o_t and enforce constraint p_t + o_t <= 1 via softmax
            gates = torch.softmax(gate_logits, dim=-1)
            p_t = gates[:, 0:1].unsqueeze(-1)  # shape: (batch_size, 1, 1)
            o_t = gates[:, 1:2].unsqueeze(-1)  # shape: (batch_size, 1, 1)
            
            # Compute S_push: shift S_prev down by 1 (place v_t at top index 0)
            S_push = torch.cat([v_t.unsqueeze(1), S_prev[:, :-1, :]], dim=1)
            
            # Compute S_pop: shift S_prev up by 1 (place zeros at bottom index)
            zeros = torch.zeros(batch_size, 1, self.stack_width, device=S_prev.device, dtype=S_prev.dtype)
            S_pop = torch.cat([S_prev[:, 1:, :], zeros], dim=1)
            
            # Update soft stack
            S = p_t * S_push + o_t * S_pop + (1.0 - p_t - o_t) * S_prev
            
            # Record stack state and logit
            stack_states_list.append(S)
            logits_step = self.fc(h)  # shape: (batch_size, vocab_size)
            logits_list.append(logits_step)
            
        if seq_len == 0:
            logits_tensor = torch.zeros(batch_size, 0, self.vocab_size, device=x.device, dtype=self.embedding.weight.dtype)
            stack_states_tensor = torch.zeros(batch_size, 0, self.stack_depth, self.stack_width, device=x.device, dtype=self.embedding.weight.dtype)
        else:
            logits_tensor = torch.stack(logits_list, dim=1)
            stack_states_tensor = torch.stack(stack_states_list, dim=1)
            
        return logits_tensor, stack_states_tensor
