import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task, generate_abc_task

def train_model(model, X, Y, epochs=200, lr=0.01, ignore_index=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if ignore_index is not None:
        loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()

def evaluate_model(model, X, Y, ignore_index=None):
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

def run_verification():
    torch.manual_seed(42)
    np.random.seed(42)
    
    vocab_size = 4
    train_len = 3
    test_len = 8
    num_samples = 150
    epochs = 250
    
    # --- COPY TASK ---
    print("\n--- COPY TASK ---")
    X_train_copy, Y_train_copy = generate_copy_task(num_samples, train_len, vocab_size)
    X_test_copy, Y_test_copy = generate_copy_task(100, test_len, vocab_size)
    
    stack_copy = StackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    dual_stack_copy = DualStackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    
    train_model(stack_copy, X_train_copy, Y_train_copy, epochs=epochs, lr=0.01)
    train_model(dual_stack_copy, X_train_copy, Y_train_copy, epochs=epochs, lr=0.01)
    
    s_tr_tok, s_tr_seq = evaluate_model(stack_copy, X_train_copy, Y_train_copy)
    s_te_tok, s_te_seq = evaluate_model(stack_copy, X_test_copy, Y_test_copy)
    
    d_tr_tok, d_tr_seq = evaluate_model(dual_stack_copy, X_train_copy, Y_train_copy)
    d_te_tok, d_te_seq = evaluate_model(dual_stack_copy, X_test_copy, Y_test_copy)
    
    print(f"StackRNN (Copy): Train Seq Acc = {s_tr_seq:.4f}, Test Seq Acc = {s_te_seq:.4f}")
    print(f"DualStackRNN (Copy): Train Seq Acc = {d_tr_seq:.4f}, Test Seq Acc = {d_te_seq:.4f}")
    
    # --- ABC TASK (deterministic n) ---
    print("\n--- ABC TASK (Fixed n) ---")
    # Using fixed n (deterministic training sequence) as in run_experiments.py
    X_train_abc, Y_train_abc = generate_abc_task(num_samples, n_max=train_len, n=train_len)
    X_test_abc, Y_test_abc = generate_abc_task(100, n_max=test_len, n=test_len)
    ignore_index = 3
    
    stack_abc = StackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    dual_stack_abc = DualStackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    
    train_model(stack_abc, X_train_abc, Y_train_abc, epochs=epochs, lr=0.01, ignore_index=ignore_index)
    train_model(dual_stack_abc, X_train_abc, Y_train_abc, epochs=epochs, lr=0.01, ignore_index=ignore_index)
    
    s_tr_tok_abc, s_tr_seq_abc = evaluate_model(stack_abc, X_train_abc, Y_train_abc, ignore_index=ignore_index)
    s_te_tok_abc, s_te_seq_abc = evaluate_model(stack_abc, X_test_abc, Y_test_abc, ignore_index=ignore_index)
    
    d_tr_tok_abc, d_tr_seq_abc = evaluate_model(dual_stack_abc, X_train_abc, Y_train_abc, ignore_index=ignore_index)
    d_te_tok_abc, d_te_seq_abc = evaluate_model(dual_stack_abc, X_test_abc, Y_test_abc, ignore_index=ignore_index)
    
    print(f"StackRNN (ABC Fixed): Train Seq Acc = {s_tr_seq_abc:.4f}, Test Seq Acc = {s_te_seq_abc:.4f}")
    print(f"DualStackRNN (ABC Fixed): Train Seq Acc = {d_tr_seq_abc:.4f}, Test Seq Acc = {d_te_seq_abc:.4f}")

    # --- ABC TASK (variable n) ---
    print("\n--- ABC TASK (Variable n) ---")
    # Training with variable n in range [1, train_len] to allow learning general rule
    X_train_abc_var, Y_train_abc_var = generate_abc_task(num_samples, n_max=train_len, n=None)
    X_test_abc_var, Y_test_abc_var = generate_abc_task(100, n_max=test_len, n=None)
    
    stack_abc_var = StackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    dual_stack_abc_var = DualStackRNN(vocab_size=vocab_size, hidden_size=32, stack_width=4, stack_depth=15)
    
    train_model(stack_abc_var, X_train_abc_var, Y_train_abc_var, epochs=epochs, lr=0.01, ignore_index=ignore_index)
    train_model(dual_stack_abc_var, X_train_abc_var, Y_train_abc_var, epochs=epochs, lr=0.01, ignore_index=ignore_index)
    
    s_tr_tok_abc_var, s_tr_seq_abc_var = evaluate_model(stack_abc_var, X_train_abc_var, Y_train_abc_var, ignore_index=ignore_index)
    s_te_tok_abc_var, s_te_seq_abc_var = evaluate_model(stack_abc_var, X_test_abc_var, Y_test_abc_var, ignore_index=ignore_index)
    
    d_tr_tok_abc_var, d_tr_seq_abc_var = evaluate_model(dual_stack_abc_var, X_train_abc_var, Y_train_abc_var, ignore_index=ignore_index)
    d_te_tok_abc_var, d_te_seq_abc_var = evaluate_model(dual_stack_abc_var, X_test_abc_var, Y_test_abc_var, ignore_index=ignore_index)
    
    print(f"StackRNN (ABC Var): Train Seq Acc = {s_tr_seq_abc_var:.4f}, Test Seq Acc = {s_te_seq_abc_var:.4f}")
    print(f"DualStackRNN (ABC Var): Train Seq Acc = {d_tr_seq_abc_var:.4f}, Test Seq Acc = {d_te_seq_abc_var:.4f}")

if __name__ == "__main__":
    run_verification()
