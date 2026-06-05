import os
import argparse
import sys
import random
import numpy as np
import torch
import torch.nn as nn

# Insert project root to sys.path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dfa import DFAGenerator
from src.models.recurrent_ssm import RecurrentSSM
from src.models.sparsity import apply_l0_mask, l0_pruning_step
from src.models.baselines import (
    CausalAttentionModel,
    Conv1DModel,
    MarkovChainModel,
    run_t_test
)

def train_model(model, X_train, Y_train, epochs=50, lr=0.03):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        loss.backward()
        optimizer.step()

def evaluate_model_accs(model, X, Y):
    with torch.no_grad():
        logits, _ = model(X)
        preds = logits.argmax(dim=-1)
        token_acc = (preds == Y).float().mean().item()
        seq_acc = (preds == Y).all(dim=-1).float().mean().item()
    return token_acc, seq_acc

def generate_alternating(length, count):
    inputs = []
    labels = []
    for i in range(count):
        if i % 2 == 0:
            seq = ['a' if j % 2 == 0 else 'b' for j in range(length)]
        else:
            seq = ['b' if j % 2 == 0 else 'a' for j in range(length)]
        inputs.append(seq)
        labels.append(seq[1:] + [seq[0]])
    
    # Convert to tensor: 0 for 'a', 1 for 'b'
    X = torch.tensor([[0 if char == 'a' else 1 for char in seq] for seq in inputs], dtype=torch.long)
    Y = torch.tensor([[0 if char == 'a' else 1 for char in label] for label in labels], dtype=torch.long)
    return X, Y

def main():
    print("Running experiments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="real_config")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Determine configuration hyperparameters
    if args.config == "mock":
        seeds = [1]
        epochs = 2
        train_len = 5
        test_len = 10
        num_samples = 5
    else:
        seeds = list(range(1, 11))
        epochs = 80
        train_len = 20
        test_len = 100
        num_samples = 100

    # Verification: Check if gradients flow back to log_alpha gate parameters
    print("Verifying gradient flow back to log_alpha gate parameters...")
    test_model = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
    # Ensure no parameter is zero so that all gates get non-zero gradients
    for p in test_model.parameters():
        p.data.add_(torch.randn_like(p.data) * 1e-2 + 1e-2)
    masked_test = apply_l0_mask(test_model)
    # Fill gates with 0.0 to prevent clamping and ensure non-zero gradient flow
    for gate in masked_test.l0_gates.values():
        gate.data.fill_(0.0)
    masked_test.train()
    optimizer_test = torch.optim.Adam(masked_test.parameters(), lr=0.03)
    loss_fn_test = nn.CrossEntropyLoss()
    x_dummy = torch.randint(0, 2, (4, 5))
    y_dummy = torch.randint(0, 2, (4, 5))
    logits_test, _ = masked_test(x_dummy)
    loss_test = loss_fn_test(logits_test.view(-1, logits_test.size(-1)), y_dummy.view(-1))
    optimizer_test.zero_grad()
    loss_test.backward()
    for name, gate in masked_test.l0_gates.items():
        assert gate.grad is not None, f"Gradient for gate {name} is None"
        assert torch.any(gate.grad != 0.0), f"Gradient for gate {name} is all zeros"
    print("Gradient flow verified successfully.")

    # 2. Setup results container
    results = {
        "SSM": {"token_accs": [], "seq_accs": [], "sparsities": []},
        "Attention": {"token_accs": [], "seq_accs": [], "sparsities": []},
        "Conv1D": {"token_accs": [], "seq_accs": [], "sparsities": []},
        "MarkovChain": {"token_accs": [], "seq_accs": [], "sparsities": []}
    }

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Generate train/val datasets
        X_train, Y_train = generate_alternating(train_len, num_samples)
        X_test, Y_test = generate_alternating(test_len, num_samples)

        # Instantiate models
        ssm = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
        att = CausalAttentionModel(vocab_size=2, d_model=8, state_dim=16)
        conv = Conv1DModel(vocab_size=2, d_model=8, state_dim=16)
        markov = MarkovChainModel(vocab_size=2, d_model=8, state_dim=16)

        # Train models
        print("Training SSM...")
        train_model(ssm, X_train, Y_train, epochs=epochs, lr=0.03)
        print("Training Attention...")
        train_model(att, X_train, Y_train, epochs=epochs, lr=0.03)
        print("Training Conv1D...")
        train_model(conv, X_train, Y_train, epochs=epochs, lr=0.03)
        print("Training MarkovChain...")
        train_model(markov, X_train, Y_train, epochs=epochs, lr=0.03)

        # Apply L0 pruning to SSM
        print("Applying L0 pruning to SSM...")
        masked_ssm = apply_l0_mask(ssm)
        l0_pruning_step(masked_ssm, temperature=0.1, target_sparsity=0.2)

        # Evaluate models on test length (length generalization)
        ssm_token, ssm_seq = evaluate_model_accs(masked_ssm, X_test, Y_test)
        ssm_sparsity = masked_ssm.pareto_frontier[-1].sparsity
        results["SSM"]["token_accs"].append(ssm_token)
        results["SSM"]["seq_accs"].append(ssm_seq)
        results["SSM"]["sparsities"].append(ssm_sparsity)

        att_token, att_seq = evaluate_model_accs(att, X_test, Y_test)
        results["Attention"]["token_accs"].append(att_token)
        results["Attention"]["seq_accs"].append(att_seq)
        results["Attention"]["sparsities"].append(0.0)

        conv_token, conv_seq = evaluate_model_accs(conv, X_test, Y_test)
        results["Conv1D"]["token_accs"].append(conv_token)
        results["Conv1D"]["seq_accs"].append(conv_seq)
        results["Conv1D"]["sparsities"].append(0.0)

        markov_token, markov_seq = evaluate_model_accs(markov, X_test, Y_test)
        results["MarkovChain"]["token_accs"].append(markov_token)
        results["MarkovChain"]["seq_accs"].append(markov_seq)
        results["MarkovChain"]["sparsities"].append(0.0)

    # 3. Print stats and run statistical significance t-test / Mann-Whitney
    print("\n================ FINAL RESULTS ================")
    for name in ["SSM", "Attention", "Conv1D", "MarkovChain"]:
        mean_token = np.mean(results[name]["token_accs"])
        mean_seq = np.mean(results[name]["seq_accs"])
        mean_sparsity = np.mean(results[name]["sparsities"])
        print(f"{name}:")
        print(f"  Token Acc: {mean_token:.4f}")
        print(f"  Seq Acc:   {mean_seq:.4f}")
        print(f"  Sparsity:  {mean_sparsity:.4f}")

    print("\n--- Statistical Significance (SSM vs Baselines) ---")
    for name in ["Attention", "Conv1D", "MarkovChain"]:
        p_t_token = run_t_test(results["SSM"]["token_accs"], results[name]["token_accs"], method="welch")
        p_mw_token = run_t_test(results["SSM"]["token_accs"], results[name]["token_accs"], method="mann_whitney")
        p_t_seq = run_t_test(results["SSM"]["seq_accs"], results[name]["seq_accs"], method="welch")
        p_mw_seq = run_t_test(results["SSM"]["seq_accs"], results[name]["seq_accs"], method="mann_whitney")
        
        print(f"SSM vs {name}:")
        print(f"  Token Acc: Welch p-val = {p_t_token:.4f}, Mann-Whitney p-val = {p_mw_token:.4f}")
        print(f"  Seq Acc:   Welch p-val = {p_t_seq:.4f}, Mann-Whitney p-val = {p_mw_seq:.4f}")

    # 4. Write the results table to results_table.csv
    table_path = os.path.join(args.output_dir, "results_table.csv")
    with open(table_path, "w") as f:
        f.write("model,accuracy,token_accuracy,sequence_accuracy,sparsity\n")
        for name in ["SSM", "Attention", "Conv1D", "MarkovChain"]:
            mean_token = np.mean(results[name]["token_accs"])
            mean_seq = np.mean(results[name]["seq_accs"])
            mean_sparsity = np.mean(results[name]["sparsities"])
            f.write(f"{name},{mean_token:.4f},{mean_token:.4f},{mean_seq:.4f},{mean_sparsity:.4f}\n")
        
    print(f"Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
