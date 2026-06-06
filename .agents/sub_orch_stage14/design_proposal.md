# Design Proposal: DualStackRNN

This document details the design and specification for the `DualStackRNN` model, which implements a Turing-complete RNN with two independent differentiable continuous stacks.

## 1. Executive Summary

A Pushdown Automaton (PDA) with a single stack is context-free, but a PDA with two independent stacks is Turing-complete. `DualStackRNN` extends the single-stack `StackRNN` logic by maintaining and updating two independent soft (differentiable) stacks. The controller RNN (a GRU cell) integrates the embedded input token and the top vectors from both stacks at each timestep, projects operations (push value, push/pop/no-op gate logits) for each stack independently, performs differentiable soft stack updates, and produces vocabulary logits.

---

## 2. Mathematical Formulation

At each sequence step $t$:

### 2.1 Inputs
- Embedded input: $x_t \in \mathbb{R}^{d_{hidden}}$ (from token sequence $X$ of shape $(B, T)$ embedded using $W_{emb}$)
- Stack 1 Top: $S^{(1)}_{t-1, 0, :} \in \mathbb{R}^{d_{width}}$
- Stack 2 Top: $S^{(2)}_{t-1, 0, :} \in \mathbb{R}^{d_{width}}$

### 2.2 Controller Recurrent Step
The cell input concatenates the current input embedding and the tops of both stacks:
$$z_t = \left[ x_t \,;\, S^{(1)}_{t-1, 0, :} \,;\, S^{(2)}_{t-1, 0, :} \right] \in \mathbb{R}^{d_{hidden} + 2 \cdot d_{width}}$$

The controller hidden state updates via GRU cell:
$$h_t = \text{GRUCell}(z_t, h_{t-1}) \in \mathbb{R}^{d_{hidden}}$$

### 2.3 Stack Projection
For each stack $i \in \{1, 2\}$, we apply a linear projection layer:
$$u^{(i)}_t = W^{(i)} h_t + b^{(i)} \in \mathbb{R}^{3 + d_{width}}$$

We extract the gate logits and push value vector:
$$g^{(i)}_t = u^{(i)}_t[:3] \in \mathbb{R}^3, \quad v^{(i)}_t = u^{(i)}_t[3:] \in \mathbb{R}^{d_{width}}$$

### 2.4 Soft Stack Updates
The gate logits are normalized using a softmax to enforce the constraint that the probabilities of push, pop, and no-op sum to 1:
$$\mathbf{a}^{(i)}_t = \text{softmax}(g^{(i)}_t) \in \mathbb{R}^3$$
- Push gate: $p^{(i)}_t = \mathbf{a}^{(i)}_t[0]$
- Pop gate: $o^{(i)}_t = \mathbf{a}^{(i)}_t[1]$
- No-op gate: $1 - p^{(i)}_t - o^{(i)}_t = \mathbf{a}^{(i)}_t[2]$

For stack $i$, we compute shifted stack states representing a push operation and a pop operation:
- **Push state** shifts the stack down by 1 and places the new value $v^{(i)}_t$ at the top (index 0):
  $$S^{(i)}_{\text{push}, t} = \left[ v^{(i)}_t \,;\, S^{(i)}_{t-1, :-1, :} \right]$$
- **Pop state** shifts the stack up by 1 and inserts zeros at the bottom:
  $$S^{(i)}_{\text{pop}, t} = \left[ S^{(i)}_{t-1, 1:, :} \,;\, \mathbf{0} \right]$$

The updated soft stack $S^{(i)}_t$ is the weighted sum of these actions:
$$S^{(i)}_t = p^{(i)}_t S^{(i)}_{\text{push}, t} + o^{(i)}_t S^{(i)}_{\text{pop}, t} + (1 - p^{(i)}_t - o^{(i)}_t) S^{(i)}_{t-1}$$

### 2.5 Vocabulary Output
The hidden state $h_t$ projects to the vocabulary logits:
$$\text{logits}_t = W_{out} h_t + b_{out} \in \mathbb{R}^{V}$$

---

## 3. Class Specification and Proposed Implementation

The class will be written in `src/models/universal_rnn.py` as follows:

```python
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
```

---

## 4. Verification and Interface Contracts

The design proposal strictly matches the contracts verified by the unit tests in `tests/test_phase_3.py`:

1. **`test_TEST_T1_F3_01_initialization`**: The model inherits from `nn.Module`.
2. **`test_TEST_T1_F3_02_forward_shape`**:
   - `logits` has shape `(batch_size, seq_len, vocab_size)`.
   - `stack_states` is a tuple/list of two tensors (`stack1_states` and `stack2_states`), each of shape `(batch_size, seq_len, stack_depth, stack_width)`.
3. **`test_TEST_T1_F3_03_stack_operations`**: Stack states contain no NaN values.
4. **`test_TEST_T1_F3_04_differentiability`**: Gradients flow to all parameters, and `param.grad` is not None.
5. **`test_TEST_T1_F3_05_stack_dimensions`**: `model.stack_width` and `model.stack_depth` are exposed on the class/instance.
6. **`test_TEST_T2_F3_01_zero_stack_width`**: Invalid dimensions (e.g. `stack_width <= 0`) throw `ValueError`.
7. **`test_TEST_T2_F3_04_device_transfer`**: Supports `.to(device)` operations and returns tensors on the correct device.
