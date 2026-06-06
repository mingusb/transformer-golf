import os
import sys
import torch
import torch.nn as nn
import numpy as np
import random

# Insert project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.context_sensitive import generate_copy_task, generate_abc_task
from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN

def train_model(model, X_train, Y_train, epochs=300, lr=0.03, ignore_index=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if ignore_index is not None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        loss.backward()
        optimizer.step()

def evaluate_model_accs(model, X, Y, ignore_index=None):
    model.eval()
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        if ignore_index is not None:
            mask = (Y != ignore_index)
            correct_tokens = (preds == Y) & mask
            token_acc = correct_tokens.sum().float().item() / max(mask.sum().float().item(), 1.0)
            seq_acc = ((preds == Y) | ~mask).all(dim=-1).float().mean().item()
        else:
            token_acc = (preds == Y).float().mean().item()
            seq_acc = (preds == Y).all(dim=-1).float().mean().item()
    return token_acc, seq_acc

def test_abc_multilen():
    print("--- Training on variable length ABC task ---")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train data: n randomly sampled from [1, 5]. Pads to length 15.
    X_train, Y_train = generate_abc_task(num_samples=200, n_max=5, n=None)
    # Test data: fixed n=8. Length 24.
    X_test, Y_test = generate_abc_task(num_samples=100, n_max=8, n=8)
    
    stack_model = StackRNN(vocab_size=4, hidden_size=16, stack_width=4, stack_depth=30)
    dual_model = DualStackRNN(vocab_size=4, hidden_size=16, stack_width=4, stack_depth=30)
    
    train_model(stack_model, X_train, Y_train, epochs=400, lr=0.02, ignore_index=3)
    train_model(dual_model, X_train, Y_train, epochs=400, lr=0.02, ignore_index=3)
    
    train_token_stack, train_seq_stack = evaluate_model_accs(stack_model, X_train, Y_train, ignore_index=3)
    train_token_dual, train_seq_dual = evaluate_model_accs(dual_model, X_train, Y_train, ignore_index=3)
    
    test_token_stack, test_seq_stack = evaluate_model_accs(stack_model, X_test, Y_test, ignore_index=3)
    test_token_dual, test_seq_dual = evaluate_model_accs(dual_model, X_test, Y_test, ignore_index=3)
    
    print(f"StackRNN:     Train Seq Acc: {train_seq_stack:.4f} | Test (n=8) Seq Acc: {test_seq_stack:.4f} (Token: {test_token_stack:.4f})")
    print(f"DualStackRNN: Train Seq Acc: {train_seq_dual:.4f} | Test (n=8) Seq Acc: {test_seq_dual:.4f} (Token: {test_token_dual:.4f})")

if __name__ == "__main__":
    test_abc_multilen()
