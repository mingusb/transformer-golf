import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader

# --- 1. The Task: Pure Induction Head (Dynamic Bigram) ---
# The sequence is a string of random tokens. The final token is a query.
# The model must find the query token earlier in the sequence and output the token
# that immediately followed it. This is the simplest form of in-context learning.

VOCAB_SIZE = 15
SEQ_LEN = 7 # 6 context tokens + 1 query token

def generate_dataset(num_samples):
    pass # Replaced by InductionDataset

import math
from scripts.task_spec import bitwise_match_logic, sequence_target_offset

class InductionDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        
        # Pre-generate dataset using dynamic logic from task_spec
        X = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN))
        Y = torch.zeros(num_samples, dtype=torch.long)
        offset = sequence_target_offset()
        
        num_bits = max(1, math.ceil(math.log2(VOCAB_SIZE)))
        def check_match(a, b):
            for i in range(num_bits):
                a_bit = (a >> i) & 1
                b_bit = (b >> i) & 1
                if not bitwise_match_logic(a_bit, b_bit):
                    return False
            return True
            
        for i in range(num_samples):
            while True:
                context = torch.rand(VOCAB_SIZE).argsort(dim=-1)[:SEQ_LEN - 1].tolist()
                query = random.randint(0, VOCAB_SIZE - 1)
                
                target = None
                for j in range(len(context) - offset):
                    if check_match(context[j], query):
                        target = context[j+offset]
                        break
                
                if target is not None:
                    X[i, :-1] = torch.tensor(context)
                    X[i, -1] = query
                    Y[i] = target
                    break
            
        self.seqs = X
        self.targets = Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]



# --- 2. The Transformer (TinyGPT) ---
class MicroGPT(nn.Module):
    def __init__(self, n_layer=2, n_head=1, n_embd=8, use_mlp=True, use_ln=True, use_bias=True):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, n_embd)
        self.wpe = nn.Embedding(SEQ_LEN, n_embd)
        self.use_mlp = use_mlp
        self.use_ln = use_ln
        
        # We will implement it manually to easily toggle MLP
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            layer = nn.ModuleDict({
                'attn': nn.MultiheadAttention(n_embd, n_head, batch_first=True, bias=use_bias)
            })
            if self.use_ln:
                layer['ln1'] = nn.LayerNorm(n_embd, elementwise_affine=use_bias)
            if self.use_mlp:
                if self.use_ln:
                    layer['ln2'] = nn.LayerNorm(n_embd, elementwise_affine=use_bias)
                layer['mlp'] = nn.Sequential(
                    nn.Linear(n_embd, 2*n_embd, bias=use_bias),
                    nn.ReLU(),
                    nn.Linear(2*n_embd, n_embd, bias=use_bias)
                )
            self.layers.append(layer)
            
        if self.use_ln:
            self.ln_f = nn.LayerNorm(n_embd, elementwise_affine=use_bias)
        self.lm_head = nn.Linear(n_embd, VOCAB_SIZE, bias=use_bias)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.wte(x) + self.wpe(pos)
        
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        for layer in self.layers:
            # Attention block
            nx = layer['ln1'](x) if self.use_ln else x
            attn_out, _ = layer['attn'](nx, nx, nx, attn_mask=mask, is_causal=True)
            x = x + attn_out
            
            # Optional MLP block
            if self.use_mlp:
                nx = layer['ln2'](x) if self.use_ln else x
                x = x + layer['mlp'](nx)
                
        if self.use_ln:
            x = self.ln_f(x)
        return self.lm_head(x[:, -1, :])

# --- 3. The 3-Layer MLP ---
class SimpleMLP(nn.Module):
    def __init__(self, n_embd=8):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, n_embd)
        self.wpe = nn.Embedding(SEQ_LEN, n_embd)
        
        # Flatten the entire sequence and pass through dense layers
        self.fc1 = nn.Linear(SEQ_LEN * n_embd, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, VOCAB_SIZE)
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.wte(x) + self.wpe(pos)
        
        # Flatten
        x = x.view(B, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- 4. Training Loop ---
class ContrastiveInductionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, context=None):
        return self.criterion(logits, targets)

# --- 4. Custom Backward Pass for Induction Head Circuit ---
def custom_backward(model, x, d_logits):
    B, T = x.shape
    D = model.lm_head.weight.shape[1]
    
    with torch.no_grad():
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        h0 = model.wte(x) + model.wpe(pos)
        
        l0_in = model.layers[0]['attn'].in_proj_weight
        q0 = h0 @ l0_in[:D, :].T
        k0 = h0 @ l0_in[D:2*D, :].T
        v0 = h0 @ l0_in[2*D:, :].T
        
        score0 = (q0 @ k0.transpose(1, 2)) / (D ** 0.5)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        score0 = score0 + mask
        a0 = torch.nn.functional.softmax(score0, dim=-1)
        o0 = (a0 @ v0) @ model.layers[0]['attn'].out_proj.weight.T
        h1 = h0 + o0
        
        l1_in = model.layers[1]['attn'].in_proj_weight
        q1 = h1 @ l1_in[:D, :].T
        k1 = h1 @ l1_in[D:2*D, :].T
        v1 = h1 @ l1_in[2*D:, :].T
        
        score1 = (q1 @ k1.transpose(1, 2)) / (D ** 0.5)
        score1 = score1 + mask
        a1 = torch.nn.functional.softmax(score1, dim=-1)
        o1 = (a1 @ v1) @ model.layers[1]['attn'].out_proj.weight.T
        h2 = h1 + o1
        
        model.lm_head.weight.grad = d_logits.transpose(0, 1) @ h2[:, -1, :]
        d_h2_last = d_logits @ model.lm_head.weight
        
        d_o1_last = d_h2_last
        model.layers[1]['attn'].out_proj.weight.grad = d_o1_last.transpose(0, 1) @ (a1[:, -1, :].unsqueeze(1) @ v1).squeeze(1)
        d_a1_last_v1 = d_o1_last @ model.layers[1]['attn'].out_proj.weight
        
        d_v1 = a1[:, -1, :].unsqueeze(2) * d_a1_last_v1.unsqueeze(1)
        d_W_v1 = (h1.transpose(1, 2) @ d_v1).sum(0).transpose(0, 1)
        
        d_a1_last = (d_a1_last_v1.unsqueeze(1) @ v1.transpose(1, 2)).squeeze(1)
        d_score1_last = a1[:, -1, :] * (d_a1_last - (a1[:, -1, :] * d_a1_last).sum(dim=1, keepdim=True)) / (D ** 0.5)
        
        d_q1_last = (d_score1_last.unsqueeze(1) @ k1).squeeze(1)
        d_W_q1 = d_q1_last.transpose(0, 1) @ h1[:, -1, :]
        
        d_k1 = d_score1_last.unsqueeze(2) * q1[:, -1, :].unsqueeze(1)
        d_W_k1 = (h1.transpose(1, 2) @ d_k1).sum(0).transpose(0, 1)
        
        model.layers[1]['attn'].in_proj_weight.grad = torch.cat([d_W_q1, d_W_k1, d_W_v1], dim=0)
        
        d_h1 = d_v1 @ l1_in[2*D:, :] + d_k1 @ l1_in[D:2*D, :]
        d_h1[:, -1, :] += d_q1_last @ l1_in[:D, :] + d_h2_last
        
        d_o0 = d_h1
        model.layers[0]['attn'].out_proj.weight.grad = (d_o0.transpose(1, 2) @ (a0 @ v0)).sum(0)
        
        d_a0_v0 = d_o0 @ model.layers[0]['attn'].out_proj.weight
        d_v0 = a0.transpose(1, 2) @ d_a0_v0
        d_W_v0 = (h0.transpose(1, 2) @ d_v0).sum(0).transpose(0, 1)
        
        d_a0 = d_a0_v0 @ v0.transpose(1, 2)
        d_score0 = a0 * (d_a0 - (a0 * d_a0).sum(dim=2, keepdim=True)) / (D ** 0.5)
        d_score0.masked_fill_(mask == float('-inf'), 0)
        
        d_q0 = d_score0 @ k0
        d_W_q0 = (h0.transpose(1, 2) @ d_q0).sum(0).transpose(0, 1)
        
        d_k0 = d_score0.transpose(1, 2) @ q0
        d_W_k0 = (h0.transpose(1, 2) @ d_k0).sum(0).transpose(0, 1)
        
        model.layers[0]['attn'].in_proj_weight.grad = torch.cat([d_W_q0, d_W_k0, d_W_v0], dim=0)
        
        d_h0 = d_h1 + d_v0 @ l0_in[2*D:, :] + d_q0 @ l0_in[:D, :] + d_k0 @ l0_in[D:2*D, :]
        
        model.wpe.weight.grad = d_h0.sum(0)
        d_wte = torch.zeros_like(model.wte.weight)
        d_wte.index_add_(0, x.view(-1), d_h0.view(-1, D))
        model.wte.weight.grad = d_wte

def train(model, name, train_dataset, eval_X_train, eval_Y_train, eval_X_test, eval_Y_test, epochs=40, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Integrate torch.compile with AOTAutograd
    if False: #hasattr(torch, 'compile'):
        # using 'inductor' backend to maximize FLOPS, and 'reduce-overhead' to minimize CPU dispatch overhead
        # under the hood, inductor uses AOTAutograd. We could also explicitly use 'aot_eager' for pure AOTAutograd.
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveInductionLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, 
                              num_workers=0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            
            if name == "GPT-PureAttn":
                logits = model(bx)
                loss = criterion(logits, by, context=bx)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                logits = model(bx)
                loss = criterion(logits, by, context=bx)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            eval_bx_train = eval_X_train.to(device, non_blocking=True)
            eval_by_train = eval_Y_train.to(device, non_blocking=True)
            train_preds = model(eval_bx_train).argmax(dim=-1)
            train_acc = (train_preds == eval_by_train).float().mean().item()
            
            eval_bx_test = eval_X_test.to(device, non_blocking=True)
            eval_by_test = eval_Y_test.to(device, non_blocking=True)
            test_preds = model(eval_bx_test).argmax(dim=-1)
            test_acc = (test_preds == eval_by_test).float().mean().item()
            
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"{name} - Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%")

# --- 5. Pure Logic Circuits ---
def get_bit(value, bit_index):
    return (value >> bit_index) & 1

import json
import os
try:
    with open("docs/z3_ast.json", "r") as f:
        z3_ast = json.load(f)
    exec(z3_ast["python_code"], globals())
except FileNotFoundError:
    def bool_eq(a, b):
        return (a & b) | ((~a & 1) & (~b & 1))

def predict_next_token(context, query):
    num_bits = 4
    ctx_bits = [[get_bit(tok, i) for i in range(num_bits)] for tok in context]
    q_bits = [get_bit(query, i) for i in range(num_bits)]

    M = []
    for j in range(len(context) - 1):
        match_j = bool_eq(ctx_bits[j][0], q_bits[0]) & bool_eq(ctx_bits[j][1], q_bits[1]) & bool_eq(ctx_bits[j][2], q_bits[2]) & bool_eq(ctx_bits[j][3], q_bits[3])
        M.append(match_j)

    y_bits = [0] * num_bits
    for i in range(num_bits):
        terms = []
        for j in range(len(context) - 1):
            terms.append(M[j] & ctx_bits[j+1][i])
        if not terms: terms = [0]
        y_bits[i] = terms[0]
        for t in terms[1:]:
            y_bits[i] |= t

    y = y_bits[0] | (y_bits[1] << 1) | (y_bits[2] << 2) | (y_bits[3] << 3)
    return y

import ctypes
import os

try:
    _lib = ctypes.CDLL(os.path.abspath("optimized_true_gpt.so"))
    _lib.predict_next_token_optimized.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64]
    _lib.predict_next_token_optimized.restype = ctypes.c_int64
except OSError:
    _lib = None

def predict_next_token_optimized(context, query):
    if _lib is None:
        raise RuntimeError("optimized_true_gpt.so not found. Please run transpile_circuit.py or build it manually.")
    
    import numpy as np
    context_arr = np.array(context, dtype=np.int64)
    ctx_ptr = context_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    return _lib.predict_next_token_optimized(ctx_ptr, len(context), query)


from functools import reduce
def predict_next_token_clojure(context, query):
    def bitwise_logic(j):
        diff = context[j] ^ query
        any_diff = (diff | (diff >> 1) | (diff >> 2) | (diff >> 3)) & 1
        mask = -(any_diff ^ 1)
        return mask & context[j+1]
        
    return reduce(lambda acc, val: acc | val, map(bitwise_logic, range(len(context) - 1)), 0)

def predict_next_token_verilog(context, query):
    y = 0
    for j in range(len(context) - 1):
        # MUX equivalent representation
        if context[j] == query:
            y |= context[j+1]
    return y

if __name__ == "__main__":
    print("Generating dataset...")
    train_dataset = InductionDataset(15000)
    test_dataset = InductionDataset(1000)

    eval_X_train, eval_Y_train = [], []
    for i in range(1500):
        x, y = train_dataset[i]
        eval_X_train.append(x); eval_Y_train.append(y)
    eval_X_train = torch.stack(eval_X_train)
    eval_Y_train = torch.stack(eval_Y_train)

    eval_X_test, eval_Y_test = [], []
    for i in range(1000):
        x, y = test_dataset[i]
        eval_X_test.append(x); eval_Y_test.append(y)
    eval_X_test = torch.stack(eval_X_test)
    eval_Y_test = torch.stack(eval_Y_test)

    print("\n--- Training 3-Layer MLP ---")
    mlp = SimpleMLP(n_embd=8)
    train(mlp, "MLP", train_dataset, eval_X_train, eval_Y_train, eval_X_test, eval_Y_test)

    print("\n--- Training MicroGPT (2-Layer Attention-Only, No LN, No Bias) ---")
    gpt = MicroGPT(n_embd=8, use_mlp=False, use_ln=False, use_bias=False)
    train(gpt, "GPT-PureAttn", train_dataset, eval_X_train, eval_Y_train, eval_X_test, eval_Y_test)

    print("\n--- Demonstration ---")
    # Show an example of how the models perform on a new random sequence
    test_seq, test_tgt = test_dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_seq_cuda = test_seq.unsqueeze(0).to(device)

    print(f"Sequence Context: {test_seq[:-1].tolist()}")
    print(f"Query Key:        {test_seq[-1].item()}")
    print(f"True Target:      {test_tgt.item()}")

    mlp.eval()
    gpt.eval()
    mlp_pred = mlp(test_seq_cuda).argmax(dim=-1).item()
    gpt_pred = gpt(test_seq_cuda).argmax(dim=-1).item()

    print(f"MLP Guessed:      {mlp_pred} {'(WRONG)' if mlp_pred != test_tgt.item() else '(CORRECT)'}")
    print(f"GPT Guessed:      {gpt_pred} {'(WRONG)' if gpt_pred != test_tgt.item() else '(CORRECT)'}")

    print("\n--- Testing Shannon N-Gram Markov Baseline ---")
    # Build a simple N-gram lookup table from train data
    markov_dict = {}
    for x, y in zip(eval_X_train, eval_Y_train):
        ctx_tuple = tuple(x.tolist())
        if ctx_tuple not in markov_dict:
            markov_dict[ctx_tuple] = []
        markov_dict[ctx_tuple].append(y.item())
    


    # Predict by picking the most common following token for the N-gram context
    shannon_correct = 0
    for x, y in zip(eval_X_test, eval_Y_test):
        ctx_tuple = tuple(x.tolist())
        if ctx_tuple in markov_dict:
            counts = {}
            for v in markov_dict[ctx_tuple]:
                counts[v] = counts.get(v, 0) + 1
            pred = max(counts, key=counts.get)
            if pred == y.item():
                shannon_correct += 1
        else:
            # Random guess if context never seen
            import random
            if random.randint(0, 15) == y.item():
                shannon_correct += 1
                
    print(f"Shannon Markov | Test Acc: {shannon_correct / len(eval_X_test) * 100:.1f}%")

    print("\n--- Testing Boolean Logic Circuit Equivalency ---")
    bool_correct = sum(predict_next_token(x[:-1].tolist(), x[-1].item()) == y.item() for x, y in zip(eval_X_test, eval_Y_test))
    print(f"Boolean Circuit | Test Acc: {bool_correct / len(eval_X_test) * 100:.1f}%")


    import json
    test_data = {"x": [x[:-1].tolist() for x in eval_X_test], "y": [y.item() for y in eval_Y_test], "q": [x[-1].item() for x in eval_X_test]}
    import os
    os.makedirs("docs", exist_ok=True)
    with open("docs/test_data.json", "w") as f:
        json.dump(test_data, f)
        
    # Save the model for empirical transpilation
    torch.save(gpt.state_dict(), "gpt.pt")
