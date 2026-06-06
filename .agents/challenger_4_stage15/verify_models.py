import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Insert project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task, generate_abc_task

def train_model(model, X, Y, epochs=150, lr=0.03, ignore_index=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
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
    
    print("==================================================")
    print("VERIFYING COPY TASK LENGTH GENERALIZATION")
    print("==================================================")
    
    # Copy task setup
    vocab_size = 6
    train_len = 10
    num_samples = 150
    epochs = 150
    
    X_train_copy, Y_train_copy = generate_copy_task(num_samples, train_len, vocab_size)
    
    # We will test on train_len (10), and generalized lengths (20, 40)
    test_lengths = [10, 20, 40]
    
    # Initialize models
    stack_rnn_copy = StackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=60)
    dual_stack_rnn_copy = DualStackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=60)
    
    print("Training StackRNN on Copy task...")
    train_model(stack_rnn_copy, X_train_copy, Y_train_copy, epochs=epochs, lr=0.02)
    print("Training DualStackRNN on Copy task...")
    train_model(dual_stack_rnn_copy, X_train_copy, Y_train_copy, epochs=epochs, lr=0.02)
    
    copy_results = []
    for t_len in test_lengths:
        X_test, Y_test = generate_copy_task(num_samples, t_len, vocab_size)
        s_tok, s_seq = evaluate_model(stack_rnn_copy, X_test, Y_test)
        d_tok, d_seq = evaluate_model(dual_stack_rnn_copy, X_test, Y_test)
        copy_results.append((t_len, s_tok, s_seq, d_tok, d_seq))
        
    print("\nCopy Task Results Table:")
    print(f"{'Length':<10} | {'StackRNN Token':<15} | {'StackRNN Seq':<15} | {'DualStackRNN Token':<18} | {'DualStackRNN Seq':<18}")
    print("-" * 80)
    for length, s_tok, s_seq, d_tok, d_seq in copy_results:
        print(f"{length:<10} | {s_tok:<15.4f} | {s_seq:<15.4f} | {d_tok:<18.4f} | {d_seq:<18.4f}")
        
    print("\n==================================================")
    print("VERIFYING ABC TASK LENGTH GENERALIZATION")
    print("==================================================")
    
    # ABC task setup
    # ignore_index is 3 (padding token)
    # We train on exactly n = 8 (sequence length 24)
    # and test on n = 8, 15, 30
    train_n = 8
    X_train_abc, Y_train_abc = generate_abc_task(num_samples, train_n, train_n)
    
    test_ns = [8, 15, 30]
    
    stack_rnn_abc = StackRNN(vocab_size=4, hidden_size=16, stack_width=4, stack_depth=60)
    dual_stack_rnn_abc = DualStackRNN(vocab_size=4, hidden_size=16, stack_width=4, stack_depth=60)
    
    print("Training StackRNN on ABC task...")
    train_model(stack_rnn_abc, X_train_abc, Y_train_abc, epochs=epochs, lr=0.02, ignore_index=3)
    print("Training DualStackRNN on ABC task...")
    train_model(dual_stack_rnn_abc, X_train_abc, Y_train_abc, epochs=epochs, lr=0.02, ignore_index=3)
    
    abc_results = []
    for t_n in test_ns:
        X_test, Y_test = generate_abc_task(num_samples, t_n, t_n)
        s_tok, s_seq = evaluate_model(stack_rnn_abc, X_test, Y_test, ignore_index=3)
        d_tok, d_seq = evaluate_model(dual_stack_rnn_abc, X_test, Y_test, ignore_index=3)
        abc_results.append((t_n, s_tok, s_seq, d_tok, d_seq))
        
    print("\nABC Task Results Table:")
    print(f"{'n':<10} | {'StackRNN Token':<15} | {'StackRNN Seq':<15} | {'DualStackRNN Token':<18} | {'DualStackRNN Seq':<18}")
    print("-" * 80)
    for n_val, s_tok, s_seq, d_tok, d_seq in abc_results:
        print(f"{n_val:<10} | {s_tok:<15.4f} | {s_seq:<15.4f} | {d_tok:<18.4f} | {d_seq:<18.4f}")

if __name__ == "__main__":
    run_verification()
