import torch
import torch.nn as nn
from typing import Tuple, Optional

class DualStackRNN(nn.Module):
    # Exposing stack_width and stack_depth as class-level type annotations / attributes
    stack_width: int
    stack_depth: int

    def __init__(self, vocab_size: int, hidden_size: int, stack_width: int, stack_depth: int):
        super().__init__()
        if vocab_size <= 0 or hidden_size <= 0 or stack_width <= 0 or stack_depth <= 0:
            raise ValueError("Dimensions must be positive")
            
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Controller recurrent cell: takes concatenated embedding + BOTH stack tops as input
        self.rnn_cell = nn.GRUCell(hidden_size + 2 * stack_width, hidden_size)
        
        # Project hidden state to gates (push, pop, noop) and push value v_t for each stack
        self.stack1_proj = nn.Linear(hidden_size, 3 + stack_width)
        self.stack2_proj = nn.Linear(hidden_size, 3 + stack_width)
        
        # Readout layer projecting hidden state to vocabulary logits
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Dual-Stack Augmented RNN.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len)
            state (torch.Tensor, optional): Initial controller hidden state of shape (batch_size, hidden_size)
            
        Returns:
            logits (torch.Tensor): Logits of shape (batch_size, seq_len, vocab_size)
            stack_states (Tuple[torch.Tensor, torch.Tensor]): Tuple of stack 1 and stack 2 states, 
                each of shape (batch_size, seq_len, stack_depth, stack_width)
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
            
        # Initialize stack representations: S1, S2 of shape (batch_size, stack_depth, stack_width)
        S1 = torch.zeros(batch_size, self.stack_depth, self.stack_width, device=x.device, dtype=self.embedding.weight.dtype)
        S2 = torch.zeros(batch_size, self.stack_depth, self.stack_width, device=x.device, dtype=self.embedding.weight.dtype)
        
        logits_list = []
        stack1_states_list = []
        stack2_states_list = []
        
        # Embed inputs
        embedded_x = self.embedding(x)  # shape: (batch_size, seq_len, hidden_size)
        
        for t in range(seq_len):
            x_t = embedded_x[:, t, :]  # shape: (batch_size, hidden_size)
            S1_prev = S1  # shape: (batch_size, stack_depth, stack_width)
            S2_prev = S2  # shape: (batch_size, stack_depth, stack_width)
            
            # Stack tops S_{t-1, 0}
            stack1_top = S1_prev[:, 0, :]  # shape: (batch_size, stack_width)
            stack2_top = S2_prev[:, 0, :]  # shape: (batch_size, stack_width)
            
            # Concatenate input and both stack tops
            cell_input = torch.cat([x_t, stack1_top, stack2_top], dim=-1)  # shape: (batch_size, hidden_size + 2 * stack_width)
            
            h = self.rnn_cell(cell_input, h)  # shape: (batch_size, hidden_size)
            
            # Stack 1 updates
            proj1 = self.stack1_proj(h)  # shape: (batch_size, 3 + stack_width)
            gate_logits1 = proj1[:, :3]
            v_t1 = proj1[:, 3:]
            
            gates1 = torch.softmax(gate_logits1, dim=-1)
            p_t1 = gates1[:, 0:1].unsqueeze(-1)  # shape: (batch_size, 1, 1)
            o_t1 = gates1[:, 1:2].unsqueeze(-1)  # shape: (batch_size, 1, 1)
            
            S_push1 = torch.cat([v_t1.unsqueeze(1), S1_prev[:, :-1, :]], dim=1)
            zeros1 = torch.zeros(batch_size, 1, self.stack_width, device=S1_prev.device, dtype=S1_prev.dtype)
            S_pop1 = torch.cat([S1_prev[:, 1:, :], zeros1], dim=1)
            S1 = p_t1 * S_push1 + o_t1 * S_pop1 + (1.0 - p_t1 - o_t1) * S1_prev
            
            # Stack 2 updates
            proj2 = self.stack2_proj(h)  # shape: (batch_size, 3 + stack_width)
            gate_logits2 = proj2[:, :3]
            v_t2 = proj2[:, 3:]
            
            gates2 = torch.softmax(gate_logits2, dim=-1)
            p_t2 = gates2[:, 0:1].unsqueeze(-1)  # shape: (batch_size, 1, 1)
            o_t2 = gates2[:, 1:2].unsqueeze(-1)  # shape: (batch_size, 1, 1)
            
            S_push2 = torch.cat([v_t2.unsqueeze(1), S2_prev[:, :-1, :]], dim=1)
            zeros2 = torch.zeros(batch_size, 1, self.stack_width, device=S2_prev.device, dtype=S2_prev.dtype)
            S_pop2 = torch.cat([S2_prev[:, 1:, :], zeros2], dim=1)
            S2 = p_t2 * S_push2 + o_t2 * S_pop2 + (1.0 - p_t2 - o_t2) * S2_prev
            
            # Record stack states and logit
            stack1_states_list.append(S1)
            stack2_states_list.append(S2)
            logits_step = self.fc(h)  # shape: (batch_size, vocab_size)
            logits_list.append(logits_step)
            
        if seq_len == 0:
            logits_tensor = torch.zeros(batch_size, 0, self.vocab_size, device=x.device, dtype=self.embedding.weight.dtype)
            stack1_states_tensor = torch.zeros(batch_size, 0, self.stack_depth, self.stack_width, device=x.device, dtype=self.embedding.weight.dtype)
            stack2_states_tensor = torch.zeros(batch_size, 0, self.stack_depth, self.stack_width, device=x.device, dtype=self.embedding.weight.dtype)
        else:
            logits_tensor = torch.stack(logits_list, dim=1)
            stack1_states_tensor = torch.stack(stack1_states_list, dim=1)
            stack2_states_tensor = torch.stack(stack2_states_list, dim=1)
            
        return logits_tensor, (stack1_states_tensor, stack2_states_tensor)
