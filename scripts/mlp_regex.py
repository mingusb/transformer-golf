import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Load Regular Expressions
DATA_PATH = 'regex_corpus.txt'
if not os.path.exists(DATA_PATH):
    print("Technical manual not found. Please generate it first.")
    exit(1)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 64
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class MLPCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query_synapse = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_synapse = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.score_bias = nn.Parameter(torch.zeros(self.head_dim))
        self.score_activation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.head_dim, 1)
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd)
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        x_heads = x.view(B, T, self.n_head, self.head_dim)
        q = self.query_synapse(x_heads)
        k = self.key_synapse(x_heads)
        
        # Optimization 1 (Dendritic Tree Minimizer): Native broadcasting instead of expand()
        q_i = q.unsqueeze(2)
        k_j = k.unsqueeze(1)
        synaptic_sum = q_i + k_j + self.score_bias
        
        scores = self.score_activation(synaptic_sum).squeeze(-1)
        scores = scores.permute(0, 3, 1, 2)
        tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~tril, float('-inf'))
        probs = F.softmax(scores, dim=-1)
        
        v = self.value_mlp(x).view(B, T, self.n_head, self.head_dim)
        v = v.permute(0, 2, 1, 3)
        out = probs @ v
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)

class MLPBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = MLPCausalSelfAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        # Optimization 1 (Dendritic Tree Minimizer): No LayerNorm!
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class MLPTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[MLPBlock(n_embd, n_head) for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        logits = self.head(x)
        
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.size()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class ConnectionistBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.rnn = nn.RNN(n_embd, n_embd, batch_first=True, nonlinearity='relu')
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        x = x + rnn_out
        x = x + self.mlp(x)
        return x

class PureConnectionistLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_layer=3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[ConnectionistBlock(n_embd) for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        x = self.blocks(x)
        logits = self.head(x)
        
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.size()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class SparseRecurrentLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, sparsity=0.8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        
        # A single, sparse recurrent population of neurons (Liquid State Machine)
        # No residuals, no MLP blocks. Just one physical layer.
        self.rnn = nn.RNN(n_embd, n_embd, batch_first=True, nonlinearity='relu')
        
        # Enforce sparsity on the recurrent weight matrix
        with torch.no_grad():
            mask = (torch.rand_like(self.rnn.weight_hh_l0) > sparsity).float()
            self.register_buffer('mask', mask)
            self.rnn.weight_hh_l0.data *= self.mask
            # Register a hook to maintain sparsity during training (zero out gradients for dropped weights)
            self.rnn.weight_hh_l0.register_hook(lambda grad: grad * self.mask)

        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx)
        x, _ = self.rnn(x)  # Single population of neurons updating over time
        logits = self.head(x)
        
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.size()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class VanillaTDLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        
        # Layer 1: Three distinct synaptic matrices for the three temporal inputs
        self.W_t2 = nn.Linear(n_embd, 4 * n_embd)
        self.W_t1 = nn.Linear(n_embd, 4 * n_embd)
        self.W_t0 = nn.Linear(n_embd, 4 * n_embd)
        
        # Layer 2: Output Projection
        self.head = nn.Linear(4 * n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx) # (B, T, C)
        
        # Pad the sequence by 2 on the time dimension to always have access to t-2 and t-1
        x_pad = F.pad(x.transpose(1, 2), (2, 0)).transpose(1, 2)
        
        # Extract the literal physical inputs for t-2, t-1, and t0
        x_t2 = x_pad[:, :-2, :]
        x_t1 = x_pad[:, 1:-1, :]
        x_t0 = x_pad[:, 2:, :]
        
        # Hidden Layer receives independent synaptic projections summed together
        # (t-2, t-1, t0) -> Hidden
        hidden = F.relu(self.W_t2(x_t2) + self.W_t1(x_t1) + self.W_t0(x_t0))
        
        # Hidden -> Output
        logits = self.head(hidden)
        
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.size()
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train():
    # 1. Train MLP-Transformer (No LayerNorm)
    model1 = MLPTransformer(vocab_size, n_embd=64).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    
    print("--- Training Pure Connectionist MLP-Transformer on Regular Expressions ---")
    for iter in range(2500):
        x, y = get_batch('train')
        logits, loss = model1(x, y)
        optimizer1.zero_grad(set_to_none=True)
        loss.backward()
        optimizer1.step()
        if iter % 500 == 0:
            print(f"Iter {iter:4d} | Train Loss: {loss.item():.4f}")
            
    print(f"Iter 2500 | Train Loss: {loss.item():.4f}\n")
    print("--- Generating Regular Expressions (MLP-Transformer) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model1.generate(context, max_new_tokens=250)
    print(decode(generated[0].tolist()))
    print("\n")

    # 2. Train Recurrent-MLP (3-Layer)
    model2 = PureConnectionistLM(vocab_size, n_embd=64).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    
    print("--- Training Pure Connectionist Recurrent-MLP LM on Regular Expressions ---")
    for iter in range(2500):
        x, y = get_batch('train')
        logits, loss = model2(x, y)
        optimizer2.zero_grad(set_to_none=True)
        loss.backward()
        optimizer2.step()
        if iter % 500 == 0:
            print(f"Iter {iter:4d} | Train Loss: {loss.item():.4f}")
            
    print(f"Iter 2500 | Train Loss: {loss.item():.4f}\n")
    print("--- Generating Regular Expressions (Recurrent-MLP) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model2.generate(context, max_new_tokens=250)
    print(decode(generated[0].tolist()))
    print("\n")
    
    # 3. Train Sparse 1-Layer Recurrent-MLP (16 Neurons, 85% Sparse)
    model3 = SparseRecurrentLM(vocab_size, n_embd=16, sparsity=0.85).to(device)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
    
    print("--- Training Pure Connectionist Sparse Recurrent-MLP on Regular Expressions ---")
    for iter in range(2500):
        x, y = get_batch('train')
        logits, loss = model3(x, y)
        optimizer3.zero_grad(set_to_none=True)
        loss.backward()
        optimizer3.step()
        if iter % 500 == 0:
            print(f"Iter {iter:4d} | Train Loss: {loss.item():.4f}")
            
    print(f"Iter 2500 | Train Loss: {loss.item():.4f}\n")
    print("--- Generating Regular Expressions (Sparse Recurrent-MLP) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model3.generate(context, max_new_tokens=250)
    print(decode(generated[0].tolist()))
    print("\n--- Physical Neural Topology (Mermaid) ---")
    print("```mermaid")
    print("graph TD")
    print("    classDef neuron fill:#f9f2e7,stroke:#d4a373,stroke-width:2px,color:#000;")
    for i in range(16):
        print(f"    N{i}((\"N_{i}\")):::neuron")
    
    weights = model3.rnn.weight_hh_l0.data.cpu().numpy()
    for i in range(16):
        for j in range(16):
            if abs(weights[i, j]) > 1e-5:
                # i is output neuron, j is input neuron. So connection is j -> i.
                weight_val = f"{weights[i, j]:.2f}"
                print(f"    N{j} -->|\"{weight_val}\"| N{i}")
    print("```\n")

    # 4. Train Vanilla TDL (Tapped Delay Line) 3-Layer Network
    model4 = VanillaTDLLM(vocab_size, n_embd=64).to(device)
    optimizer4 = torch.optim.AdamW(model4.parameters(), lr=1e-3)
    
    print("--- Training Pure Connectionist Vanilla TDL on Regular Expressions ---")
    for iter in range(2500):
        x, y = get_batch('train')
        logits, loss = model4(x, y)
        optimizer4.zero_grad(set_to_none=True)
        loss.backward()
        optimizer4.step()
        if iter % 500 == 0:
            print(f"Iter {iter:4d} | Train Loss: {loss.item():.4f}")
            
    print(f"Iter 2500 | Train Loss: {loss.item():.4f}\n")
    print("--- Generating Regular Expressions (Vanilla TDL) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model4.generate(context, max_new_tokens=250)
    print(decode(generated[0].tolist()))

    # 5. Train Vanilla TDL using 1990s Vanilla SGD
    model5 = VanillaTDLLM(vocab_size, n_embd=64).to(device)
    optimizer5 = torch.optim.SGD(model5.parameters(), lr=0.1)
    
    print("--- Training Pure Connectionist Vanilla TDL (1990s SGD) on Regular Expressions ---")
    for iter in range(2500):
        x, y = get_batch('train')
        logits, loss = model5(x, y)
        optimizer5.zero_grad(set_to_none=True)
        loss.backward()
        optimizer5.step()
        if iter % 500 == 0:
            print(f"Iter {iter:4d} | Train Loss: {loss.item():.4f}")
            
    print(f"Iter 2500 | Train Loss: {loss.item():.4f}\n")
    print("--- Generating Regular Expressions (Vanilla TDL - 1990s SGD) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model5.generate(context, max_new_tokens=250)
    print(decode(generated[0].tolist()))

if __name__ == '__main__':
    train()
