import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import random
from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task

def train_multi_lengths(model, epochs=400):
    vocab_size = 4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Alternate train lengths: 1, 2, 3
        train_len = random.choice([2, 3])
        X, Y = generate_copy_task(64, train_len, vocab_size)
        logits, _ = model(X)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()

def eval_length(model, length):
    model.eval()
    vocab_size = 4
    X, Y = generate_copy_task(100, length, vocab_size)
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        # We only care about the copied part (after the delimiter)
        # The delimiter is at index `length`
        # The copied tokens are from index `length + 1` to `2 * length`
        correct = preds[:, length+1:] == Y[:, length+1:]
        token_acc = correct.float().mean().item()
        seq_acc = correct.all(dim=-1).float().mean().item()
    return token_acc, seq_acc

print("=== Training StackRNN ===")
stack_rnn = StackRNN(vocab_size=4, hidden_size=32, stack_width=4, stack_depth=10)
train_multi_lengths(stack_rnn, epochs=500)
for l in [2, 3, 4, 5]:
    t_acc, s_acc = eval_length(stack_rnn, l)
    print(f"Len {l}: Token Acc = {t_acc:.4f}, Seq Acc = {s_acc:.4f}")

print("\n=== Training DualStackRNN ===")
dual_stack_rnn = DualStackRNN(vocab_size=4, hidden_size=32, stack_width=4, stack_depth=10)
train_multi_lengths(dual_stack_rnn, epochs=500)
for l in [2, 3, 4, 5]:
    t_acc, s_acc = eval_length(dual_stack_rnn, l)
    print(f"Len {l}: Token Acc = {t_acc:.4f}, Seq Acc = {s_acc:.4f}")
