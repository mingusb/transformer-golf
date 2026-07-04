import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import random
from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task

def generate_variable_length_batch(num_samples, min_len, max_len, vocab_size):
    # Since inputs must be batched, we pad shorter sequences with 0 at the end.
    # But wait! Padding with 0 at the end can be confusing.
    # Alternatively, we can generate a batch where all sequences in the batch have the same length,
    # but the length varies from batch to batch. This is much cleaner!
    length = random.randint(min_len, max_len)
    return generate_copy_task(num_samples, length, vocab_size), length

def train_and_eval(model, epochs=800):
    vocab_size = 4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        (X, Y), length = generate_variable_length_batch(128, 2, 6, vocab_size)
        logits, _ = model(X)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
        loss.backward()
        optimizer.step()
        
    # Evaluate generalization on length 10
    model.eval()
    test_len = 10
    X_test, Y_test = generate_copy_task(100, test_len, vocab_size)
    with torch.no_grad():
        logits, _ = model(X_test)
        preds = logits.argmax(dim=-1)
        # Check token accuracy on the copied part
        copied_correct = preds[:, test_len+1:] == Y_test[:, test_len+1:]
        token_acc = copied_correct.float().mean().item()
        seq_acc = copied_correct.all(dim=-1).float().mean().item()
        
    return token_acc, seq_acc

print("=== STACK RNN ===")
torch.manual_seed(42)
stack_rnn = StackRNN(vocab_size=4, hidden_size=32, stack_width=4, stack_depth=20)
stack_tok, stack_seq = train_and_eval(stack_rnn, epochs=800)
print(f"StackRNN on Test Length 10 - Token Acc: {stack_tok:.4f}, Seq Acc: {stack_seq:.4f}")

print("\n=== DUAL STACK RNN ===")
torch.manual_seed(42)
dual_stack_rnn = DualStackRNN(vocab_size=4, hidden_size=32, stack_width=4, stack_depth=20)
dual_tok, dual_seq = train_and_eval(dual_stack_rnn, epochs=800)
print(f"DualStackRNN on Test Length 10 - Token Acc: {dual_tok:.4f}, Seq Acc: {dual_seq:.4f}")
