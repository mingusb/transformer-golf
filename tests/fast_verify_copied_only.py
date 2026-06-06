import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task

def train_model(model, X, Y, epochs=300, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()

def evaluate_copy_portion(model, length, num_samples=100):
    model.eval()
    vocab_size = 4
    X, Y = generate_copy_task(num_samples, length, vocab_size)
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        # Copied portion starts after the delimiter (index length)
        # So we check indices from length + 1 to 2 * length
        correct = preds[:, length+1:] == Y[:, length+1:]
        token_acc = correct.float().mean().item()
        seq_acc = correct.all(dim=-1).float().mean().item()
    return token_acc, seq_acc

def run_verification():
    torch.manual_seed(42)
    np.random.seed(42)
    
    vocab_size = 4
    train_len = 3
    num_samples = 150
    epochs = 400
    
    X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
    
    stack_copy = StackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    dual_stack_copy = DualStackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    
    print("Training StackRNN...")
    train_model(stack_copy, X_train, Y_train, epochs=epochs, lr=0.01)
    
    print("Training DualStackRNN...")
    train_model(dual_stack_copy, X_train, Y_train, epochs=epochs, lr=0.01)
    
    print("\n--- RESULTS (Copied Portion Only) ---")
    for l in [3, 4, 5, 6, 8]:
        s_tok, s_seq = evaluate_copy_portion(stack_copy, l)
        d_tok, d_seq = evaluate_copy_portion(dual_stack_copy, l)
        print(f"Length {l}:")
        print(f"  StackRNN    -> Token Acc: {s_tok:.4f}, Seq Acc: {s_seq:.4f}")
        print(f"  DualStackRNN -> Token Acc: {d_tok:.4f}, Seq Acc: {d_seq:.4f}")

if __name__ == "__main__":
    run_verification()
