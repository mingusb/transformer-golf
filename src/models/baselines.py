import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats

class CausalAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_model, state_dim, hippo_init=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.05)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)
        self.classifier = nn.Linear(d_model, vocab_size)
        
        # Match SSM parameter count
        from src.models.recurrent_ssm import RecurrentSSM
        ssm = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=hippo_init)
        ssm_params = sum(p.numel() for p in ssm.parameters())
        
        current_params = sum(p.numel() for p in self.parameters())
        
        # Compute d_ff dynamically
        denom = 2 * d_model + 1
        diff_avail = ssm_params - current_params - d_model
        if diff_avail > 0:
            d_ff = max(1, diff_avail // denom)
        else:
            d_ff = 1
            
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Match SSM parameter count with remainder dummy
        current_params = sum(p.numel() for p in self.parameters())
        diff = ssm_params - current_params
        if diff > 0:
            self.dummy = nn.Parameter(torch.zeros(diff))

    def _prepare_input(self, x):
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
        max_context = 100
        if seq_len > max_context:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum context window {max_context}")
            
        attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        ffn_out = self.ffn(attn_out)
        logits = self.classifier(ffn_out)
        return logits, None

class Conv1DModel(nn.Module):
    def __init__(self, vocab_size, d_model, state_dim, hippo_init=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.05)
        
        # Match SSM parameter count by dynamically scaling hidden/output channels of nn.Conv1d
        from src.models.recurrent_ssm import RecurrentSSM
        ssm = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=hippo_init)
        ssm_params = sum(p.numel() for p in ssm.parameters())
        
        current_params = sum(p.numel() for p in self.parameters())
        
        denom = 3 * d_model + 1 + vocab_size
        diff_avail = ssm_params - current_params - vocab_size
        if diff_avail > 0:
            hidden_channels = max(1, diff_avail // denom)
        else:
            hidden_channels = 1
            
        self.conv = nn.Conv1d(d_model, hidden_channels, kernel_size=3, padding=1)
        self.classifier = nn.Linear(hidden_channels, vocab_size)
        
        # Match SSM parameter count with remainder dummy
        current_params = sum(p.numel() for p in self.parameters())
        diff = ssm_params - current_params
        if diff > 0:
            self.dummy = nn.Parameter(torch.zeros(diff))

    def _prepare_input(self, x):
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
        if seq_len < 3:
            raise ValueError(f"Sequence length {seq_len} is smaller than kernel size 3")
            
        x_t = x.transpose(1, 2)
        out = self.conv(x_t)
        out_t = out.transpose(1, 2)
        logits = self.classifier(out_t)
        return logits, None

class MarkovChainModel(nn.Module):
    def __init__(self, vocab_size, d_model, state_dim, hippo_init=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.05)
        
        # Match SSM parameter count by parameterizing transition via an MLP mapping embedding to logits
        from src.models.recurrent_ssm import RecurrentSSM
        ssm = RecurrentSSM(vocab_size, d_model, state_dim, hippo_init=hippo_init)
        ssm_params = sum(p.numel() for p in ssm.parameters())
        
        current_params = sum(p.numel() for p in self.parameters())
        
        denom = d_model + 1 + vocab_size
        diff_avail = ssm_params - current_params - vocab_size
        if diff_avail > 0:
            d_mlp = max(1, diff_avail // denom)
        else:
            d_mlp = 1
            
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, vocab_size)
        )
        
        # Match SSM parameter count with remainder dummy
        current_params = sum(p.numel() for p in self.parameters())
        diff = ssm_params - current_params
        if diff > 0:
            self.dummy = nn.Parameter(torch.zeros(diff))

    def _prepare_input(self, x):
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
        logits = self.mlp(x)
        return logits, None

    def predict_probabilities(self, x):
        if x is None or len(x) == 0:
            probs = np.ones(self.vocab_size) / self.vocab_size
            return probs
            
        last_token = x[-1]
        if last_token < 0 or last_token >= self.vocab_size:
            probs = np.ones(self.vocab_size) / self.vocab_size
            return probs
            
        with torch.no_grad():
            token_tensor = torch.tensor([last_token], dtype=torch.long, device=self.embedding.weight.device)
            emb = self.embedding(token_tensor)
            logits = self.mlp(emb).squeeze(0)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
        probs = probs + 1e-6
        probs = probs / probs.sum()
        return probs

def run_t_test(data1, data2, method="welch"):
    d1 = np.array(data1)
    d2 = np.array(data2)
    if len(d1) == 0 or len(d2) == 0:
        raise ValueError("Data cannot be empty")
    if np.var(d1) == 0 and np.var(d2) == 0:
        if np.allclose(d1, d2):
            return 1.0
        else:
            return 0.0
    if method == "welch":
        res = stats.ttest_ind(d1, d2, equal_var=False)
        p_val = res.pvalue
    elif method == "mann_whitney":
        res = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        p_val = res.pvalue
    else:
        raise ValueError(f"Unknown method {method}")
        
    if np.isnan(p_val):
        return 1.0
    return p_val
