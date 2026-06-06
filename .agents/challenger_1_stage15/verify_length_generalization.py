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

def train_model(model, X_train, Y_train, epochs=150, lr=0.03, ignore_index=None):
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

def run_experiment(task_name, train_len, test_len, vocab_size, epochs=200, seeds=[1, 2, 3]):
    print(f"\n================ Running {task_name.upper()} Task ================")
    print(f"Train length: {train_len}, Test length: {test_len}")
    
    stack_token_accs = []
    stack_seq_accs = []
    dual_token_accs = []
    dual_seq_accs = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate datasets
        if task_name == "copy":
            X_train, Y_train = generate_copy_task(num_samples=100, length=train_len, vocab_size=vocab_size)
            X_test, Y_test = generate_copy_task(num_samples=100, length=test_len, vocab_size=vocab_size)
            ignore_index = None
        elif task_name == "abc":
            X_train, Y_train = generate_abc_task(num_samples=100, n_max=train_len, n=train_len)
            X_test, Y_test = generate_abc_task(num_samples=100, n_max=test_len, n=test_len)
            ignore_index = 3
        else:
            raise ValueError(f"Unknown task {task_name}")
            
        # Initialize models
        stack_model = StackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=55)
        dual_model = DualStackRNN(vocab_size=vocab_size, hidden_size=16, stack_width=4, stack_depth=55)
        
        # Train StackRNN
        train_model(stack_model, X_train, Y_train, epochs=epochs, lr=0.03, ignore_index=ignore_index)
        # Train DualStackRNN
        train_model(dual_model, X_train, Y_train, epochs=epochs, lr=0.03, ignore_index=ignore_index)
        
        # Evaluate on training length (convergence check)
        train_token_stack, train_seq_stack = evaluate_model_accs(stack_model, X_train, Y_train, ignore_index=ignore_index)
        train_token_dual, train_seq_dual = evaluate_model_accs(dual_model, X_train, Y_train, ignore_index=ignore_index)
        
        # Evaluate on testing length (generalization check)
        test_token_stack, test_seq_stack = evaluate_model_accs(stack_model, X_test, Y_test, ignore_index=ignore_index)
        test_token_dual, test_seq_dual = evaluate_model_accs(dual_model, X_test, Y_test, ignore_index=ignore_index)
        
        print(f"Seed {seed}:")
        print(f"  StackRNN      -> Train Seq Acc: {train_seq_stack:.4f}, Test Seq Acc: {test_seq_stack:.4f} (Token: {test_token_stack:.4f})")
        print(f"  DualStackRNN  -> Train Seq Acc: {train_seq_dual:.4f}, Test Seq Acc: {test_seq_dual:.4f} (Token: {test_token_dual:.4f})")
        
        stack_token_accs.append(test_token_stack)
        stack_seq_accs.append(test_seq_stack)
        dual_token_accs.append(test_token_dual)
        dual_seq_accs.append(test_seq_dual)
        
    print(f"\n--- {task_name.upper()} Summary (Averages over {len(seeds)} seeds) ---")
    print(f"StackRNN     - Test Token Acc: {np.mean(stack_token_accs):.4f}, Test Seq Acc: {np.mean(stack_seq_accs):.4f}")
    print(f"DualStackRNN - Test Token Acc: {np.mean(dual_token_accs):.4f}, Test Seq Acc: {np.mean(dual_seq_accs):.4f}")
    
    return {
        "stack_seq": np.mean(stack_seq_accs),
        "dual_seq": np.mean(dual_seq_accs)
    }

if __name__ == "__main__":
    # Run a fast copy task: train length 10, test length 30
    copy_results = run_experiment("copy", train_len=10, test_len=30, vocab_size=6, epochs=150, seeds=[1, 2])
    
    # Run a fast abc task: train length 10, test length 30
    abc_results = run_experiment("abc", train_len=10, test_len=30, vocab_size=4, epochs=150, seeds=[1, 2])
    
    # Assertions to verify copy task length generalization behavior
    # StackRNN should fail to generalize (Seq Acc close to 0)
    # DualStackRNN should show superior performance on copy generalization
    print("\n--- Length Generalization Assertion ---")
    print(f"Copy task: StackRNN Seq Acc = {copy_results['stack_seq']:.4f}, DualStackRNN Seq Acc = {copy_results['dual_seq']:.4f}")
    print(f"ABC task:  StackRNN Seq Acc = {abc_results['stack_seq']:.4f}, DualStackRNN Seq Acc = {abc_results['dual_seq']:.4f}")
    
    assert copy_results["stack_seq"] < 0.1, "StackRNN should fail on copy task length generalization"
    assert copy_results["dual_seq"] > 0.5, "DualStackRNN should succeed on copy task length generalization"
    assert abc_results["stack_seq"] < 0.1, "StackRNN should fail on abc task length generalization"
    assert abc_results["dual_seq"] > 0.5, "DualStackRNN should succeed on abc task length generalization"
    print("SUCCESS: Length generalization verification passed!")
