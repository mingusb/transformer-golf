import torch
import torch.nn as nn
import numpy as np

def make_hippo(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    A = np.where(i > j, -np.sqrt((2 * i + 1) * (2 * j + 1)), np.where(i == j, -(i + 1), 0.0))
    return A

class RecurrentSSM(nn.Module):
    def __init__(self, vocab_size, d_model, state_dim, hippo_init=True):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or state_dim <= 0:
            raise ValueError("Dimensions must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim
        self.hippo_init = hippo_init
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.05)
        
        if hippo_init:
            A_hippo = make_hippo(state_dim)
            I = np.eye(state_dim)
            A_discrete = np.linalg.inv(I - 0.5 * A_hippo) @ (I + 0.5 * A_hippo)
            self.A = nn.Parameter(torch.tensor(A_discrete, dtype=torch.float32))
        else:
            self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.05)
            
        self.B = nn.Parameter(torch.randn(state_dim, d_model) * 0.05)
        self.C = nn.Parameter(torch.randn(vocab_size, state_dim) * 0.05)
        
        # Data-dependent gating parameters
        self.gate_w = nn.Parameter(torch.randn(state_dim, d_model) * 0.05)
        self.gate_b = nn.Parameter(torch.zeros(state_dim))
        
    def _prepare_input(self, x):
        if x.dim() not in (2, 3):
            raise ValueError(f"Unsupported input dimension {x.dim()}. Expected 2 (batch, seq_len) or 3 (batch, seq_len, d_model).")
        if x.dim() == 2:
            if (x >= self.vocab_size).any() or (x < 0).any():
                raise IndexError("Token index out of vocabulary bounds")
            x = self.embedding(x)
        if x.dim() == 3 and x.shape[2] != self.d_model:
            if x.shape[2] < self.d_model:
                x = torch.nn.functional.pad(x, (0, self.d_model - x.shape[2]))
            else:
                x = x[:, :, :self.d_model]
        return x

    def forward(self, x, state=None):
        x = self._prepare_input(x)
        batch, seq_len, d_model = x.shape
        if state is not None:
            if state.shape[0] != batch or state.shape[1] != self.state_dim:
                raise RuntimeError(f"State shape size mismatch: expected {(batch, self.state_dim)}, got {state.shape}")
            if state.device != x.device:
                raise RuntimeError(f"State device {state.device} does not match input device {x.device}")
        if state is None:
            state = torch.zeros(batch, self.state_dim, device=x.device)
            
        logits = []
        current_state = state
        
        for t in range(seq_len):
            xt = x[:, t, :]
            gate = torch.sigmoid(xt @ self.gate_w.T + self.gate_b)
            proposed_state = torch.tanh(current_state @ self.A.T + xt @ self.B.T)
            next_state = (1.0 - gate) * current_state + gate * proposed_state
            
            lt = next_state @ self.C.T
            logits.append(lt)
            current_state = next_state
            
        if seq_len == 0:
            logits_tensor = torch.zeros(batch, 0, self.vocab_size, device=x.device)
        else:
            logits_tensor = torch.stack(logits, dim=1)
            
        return logits_tensor, current_state


def train_model_tbptt(model, X_train, Y_train, chunk_len=5, epochs=50, lr=0.03):
    if chunk_len <= 0:
        raise ValueError("chunk_len must be greater than 0")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        state = None
        seq_len = X_train.shape[1]
        
        for t in range(0, seq_len, chunk_len):
            optimizer.zero_grad()
            X_chunk = X_train[:, t : t + chunk_len]
            Y_chunk = Y_train[:, t : t + chunk_len]
            
            if state is not None:
                state = state.detach()
                
            logits, next_state = model(X_chunk, state)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y_chunk.reshape(-1))
            loss.backward()
            optimizer.step()
            
            state = next_state
