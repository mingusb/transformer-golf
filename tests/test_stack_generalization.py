import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pytest
from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task, generate_abc_task

def train_model(model, X, Y, epochs=100, lr=0.02, ignore_index=None):
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

def test_copy_generalization():
    torch.manual_seed(42)
    vocab_size = 4
    train_len = 5
    test_len = 15
    num_samples = 100
    epochs = 150

    X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
    X_test, Y_test = generate_copy_task(num_samples, test_len, vocab_size)

    # Initialize models
    stack_rnn = StackRNN(vocab_size=vocab_size, hidden_size=24, stack_width=4, stack_depth=30)
    dual_stack_rnn = DualStackRNN(vocab_size=vocab_size, hidden_size=24, stack_width=4, stack_depth=30)

    print("\nTraining StackRNN on Copy task...")
    train_model(stack_rnn, X_train, Y_train, epochs=epochs, lr=0.02)
    stack_train_token, stack_train_seq = evaluate_model(stack_rnn, X_train, Y_train)
    stack_test_token, stack_test_seq = evaluate_model(stack_rnn, X_test, Y_test)

    print("Training DualStackRNN on Copy task...")
    train_model(dual_stack_rnn, X_train, Y_train, epochs=epochs, lr=0.02)
    dual_train_token, dual_train_seq = evaluate_model(dual_stack_rnn, X_train, Y_train)
    dual_test_token, dual_test_seq = evaluate_model(dual_stack_rnn, X_test, Y_test)

    print(f"Copy Task Results:")
    print(f"StackRNN - Train Token Acc: {stack_train_token:.4f}, Train Seq Acc: {stack_train_seq:.4f}")
    print(f"StackRNN - Test Token Acc: {stack_test_token:.4f}, Test Seq Acc: {stack_test_seq:.4f}")
    print(f"DualStackRNN - Train Token Acc: {dual_train_token:.4f}, Train Seq Acc: {dual_train_seq:.4f}")
    print(f"DualStackRNN - Test Token Acc: {dual_test_token:.4f}, Test Seq Acc: {dual_test_seq:.4f}")

    # DualStackRNN should generalize much better (achieving high sequence/token accuracy)
    # while StackRNN should fail to generalize to the longer length.
    assert dual_test_token > stack_test_token
    assert dual_test_seq >= stack_test_seq

def test_abc_generalization():
    torch.manual_seed(42)
    vocab_size = 4  # 0: a, 1: b, 2: c, 3: pad
    train_len = 5
    test_len = 15
    num_samples = 100
    epochs = 150
    ignore_index = 3

    X_train, Y_train = generate_abc_task(num_samples, n_max=train_len, n=None)
    X_test, Y_test = generate_abc_task(num_samples, n_max=test_len, n=test_len)

    # Initialize models
    stack_rnn = StackRNN(vocab_size=vocab_size, hidden_size=24, stack_width=4, stack_depth=30)
    dual_stack_rnn = DualStackRNN(vocab_size=vocab_size, hidden_size=24, stack_width=4, stack_depth=30)

    print("\nTraining StackRNN on ABC task...")
    train_model(stack_rnn, X_train, Y_train, epochs=epochs, lr=0.02, ignore_index=ignore_index)
    stack_train_token, stack_train_seq = evaluate_model(stack_rnn, X_train, Y_train, ignore_index=ignore_index)
    stack_test_token, stack_test_seq = evaluate_model(stack_rnn, X_test, Y_test, ignore_index=ignore_index)

    print("Training DualStackRNN on ABC task...")
    train_model(dual_stack_rnn, X_train, Y_train, epochs=epochs, lr=0.02, ignore_index=ignore_index)
    dual_train_token, dual_train_seq = evaluate_model(dual_stack_rnn, X_train, Y_train, ignore_index=ignore_index)
    dual_test_token, dual_test_seq = evaluate_model(dual_stack_rnn, X_test, Y_test, ignore_index=ignore_index)

    print(f"ABC Task Results:")
    print(f"StackRNN - Train Token Acc: {stack_train_token:.4f}, Train Seq Acc: {stack_train_seq:.4f}")
    print(f"StackRNN - Test Token Acc: {stack_test_token:.4f}, Test Seq Acc: {stack_test_seq:.4f}")
    print(f"DualStackRNN - Train Token Acc: {dual_train_token:.4f}, Train Seq Acc: {dual_train_seq:.4f}")
    print(f"DualStackRNN - Test Token Acc: {dual_test_token:.4f}, Test Seq Acc: {dual_test_seq:.4f}")

    # DualStackRNN should generalize much better (achieving high sequence/token accuracy)
    # while StackRNN should fail to generalize to the longer length.
    assert dual_test_token > stack_test_token
    assert dual_test_seq >= stack_test_seq

if __name__ == "__main__":
    test_copy_generalization()
    test_abc_generalization()
