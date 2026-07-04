import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.models.stack_rnn import StackRNN
from src.models.universal_rnn import DualStackRNN
from src.data.context_sensitive import generate_copy_task

def train_and_eval(model, train_len, epochs=500):
    vocab_size = 4
    num_samples = 100
    X_train, Y_train = generate_copy_task(num_samples, train_len, vocab_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_logits, _ = model(X_train)
                train_preds = train_logits.argmax(dim=-1)
                # We only check correctness of the copied portion
                correct = train_preds[:, train_len+1:] == Y_train[:, train_len+1:]
                train_seq_acc = correct.all(dim=-1).float().mean().item()
                print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Train Copy Seq Acc: {train_seq_acc:.4f}")
                if train_seq_acc == 1.0:
                    break

print("=== STACK RNN ===")
stack_rnn = StackRNN(vocab_size=4, hidden_size=32, stack_width=4, stack_depth=15)
train_and_eval(stack_rnn, train_len=6, epochs=500)

print("\n=== DUAL STACK RNN ===")
dual_stack_rnn = DualStackRNN(vocab_size=4, hidden_size=32, stack_width=4, stack_depth=15)
train_and_eval(dual_stack_rnn, train_len=6, epochs=500)
