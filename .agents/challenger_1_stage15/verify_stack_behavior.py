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

def train_model(model, X_train, Y_train, epochs=300, lr=0.02, ignore_index=None):
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

def test_hyperparameters(task, train_len, test_len, vocab_size, num_samples, epochs):
    print(f"\n--- Testing Task: {task.upper()} | Train Len: {train_len} | Test Len: {test_len} ---")
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if task == "copy":
        X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
        X_test, Y_test = generate_copy_task(num_samples, test_len, vocab_size)
        ignore_index = None
    elif task == "abc":
        X_train, Y_train = generate_abc_task(num_samples, n_max=train_len, n=train_len)
        X_test, Y_test = generate_abc_task(num_samples, n_max=test_len, n=test_len)
        ignore_index = 3
        
    stack_model = StackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=30)
    dual_model = DualStackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=30)
    
    train_model(stack_model, X_train, Y_train, epochs=epochs, lr=0.02, ignore_index=ignore_index)
    train_model(dual_model, X_train, Y_train, epochs=epochs, lr=0.02, ignore_index=ignore_index)
    
    train_token_stack, train_seq_stack = evaluate_model_accs(stack_model, X_train, Y_train, ignore_index=ignore_index)
    train_token_dual, train_seq_dual = evaluate_model_accs(dual_model, X_train, Y_train, ignore_index=ignore_index)
    
    test_token_stack, test_seq_stack = evaluate_model_accs(stack_model, X_test, Y_test, ignore_index=ignore_index)
    test_token_dual, test_seq_dual = evaluate_model_accs(dual_model, X_test, Y_test, ignore_index=ignore_index)
    
    print(f"StackRNN:     Train Seq Acc: {train_seq_stack:.4f} | Test Seq Acc: {test_seq_stack:.4f}")
    print(f"DualStackRNN: Train Seq Acc: {train_seq_dual:.4f} | Test Seq Acc: {test_seq_dual:.4f}")
    
    return {
        "stack_train_seq": train_seq_stack,
        "stack_test_seq": test_seq_stack,
        "dual_train_seq": train_seq_dual,
        "dual_test_seq": test_seq_dual,
    }

if __name__ == "__main__":
    # Test copy task with small lengths
    # We use num_samples=16 to ensure training convergence is feasible within a reasonable number of epochs
    res_copy = test_hyperparameters("copy", train_len=3, test_len=8, vocab_size=4, num_samples=16, epochs=400)
    
    # Test abc task with small lengths
    res_abc = test_hyperparameters("abc", train_len=3, test_len=8, vocab_size=4, num_samples=16, epochs=400)
