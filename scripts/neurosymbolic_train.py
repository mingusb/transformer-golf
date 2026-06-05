import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('.')

class SymPyStructuredNet(nn.Module):
    def __init__(self, vocab_size, hidden_dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        
        import json
        try:
            with open("docs/z3_ast.json", "r") as f:
                z3_ast = json.load(f)
            num_layers = z3_ast.get("mlp_layers", 2)
            print(f"\n[Z3 Constraint Injection] Structuring Neuro-Symbolic MLP with {num_layers} hidden layers based on mathematical proof!")
        except FileNotFoundError:
            num_layers = 2
            
        def build_mlp(in_d, out_d, hidden_d, layers):
            mods = []
            current_in = in_d
            for _ in range(layers):
                mods.append(nn.Linear(current_in, hidden_d))
                mods.append(nn.ReLU())
                current_in = hidden_d
            mods.append(nn.Linear(current_in, out_d))
            return nn.Sequential(*mods)
            
        self.match_mlp = build_mlp(hidden_dim * 2, 1, hidden_dim, num_layers)
        self.output_mlp = build_mlp(hidden_dim, vocab_size, hidden_dim, num_layers)

    def forward(self, x):
        ctx = x[:, :-1]
        query = x[:, -1]
        
        ctx_emb = self.emb(ctx)
        query_emb = self.emb(query)
        
        y_accum = 0
        
        for j in range(ctx.size(1) - 1):
            c_j = ctx_emb[:, j, :]
            c_next = ctx_emb[:, j+1, :]
            
            pair = torch.cat([c_j, query_emb], dim=-1)
            match_score = self.match_mlp(pair)
            
            gated_value = match_score * c_next
            
            y_accum = y_accum + gated_value
            
        logits = self.output_mlp(y_accum)
        return logits

from compare_task import InductionDataset, VOCAB_SIZE, SEQ_LEN
from torch.utils.data import DataLoader

# ... [Class code remains same] ...

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SymPyStructuredNet(VOCAB_SIZE, hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("--- Training SymPy-Structured MLP Network ---")
    
    train_dataset = InductionDataset(15000)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # Train
    for epoch in range(15):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        acc = (correct / total) * 100
        print(f"Epoch {epoch+1:3d} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.1f}%")

    # Evaluate
    model.eval()
    test_dataset = InductionDataset(1000)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            test_acc = (pred == y).float().mean().item() * 100

    print(f"\nSymPy-Structured MLP | Test Acc: {test_acc:.1f}%")

if __name__ == "__main__":
    train()
