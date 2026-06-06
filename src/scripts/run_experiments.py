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
from src.data.nested_brackets import generate_nested_brackets
from src.models.recurrent_ssm import RecurrentSSM
from src.models.sparsity import apply_l0_mask, l0_pruning_step
from src.models.baselines import (
    CausalAttentionModel,
    Conv1DModel,
    MarkovChainModel,
    run_t_test
)

try:
    from src.models.stack_rnn import StackRNN
    HAS_STACK_RNN = True
except ImportError:
    HAS_STACK_RNN = False

try:
    from src.models.lsm import LiquidStateMachine
    HAS_LSM = True
except ImportError:
    HAS_LSM = False


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
    parser.add_argument("--task", type=str, choices=["alternating", "nesting"], default="alternating")
    parser.add_argument("--model", type=str, default="all", choices=["all", "ssm", "attention", "conv1d", "markov", "stack_rnn", "lsm"])
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task == "nesting":
        print("Running Nesting Depth vs Accuracy Profiling experiment...")
        if args.config == "mock":
            seeds = [1]
            epochs = 2
            num_samples = 5
            eval_depths = list(range(1, 6))
        else:
            seeds = list(range(1, 11))
            epochs = 80
            num_samples = 100
            eval_depths = list(range(1, 51))

        # Determine which models to run
        if args.model == "all":
            models_to_run = ["SSM", "Attention"]
            if HAS_STACK_RNN:
                models_to_run.append("StackRNN")
            if HAS_LSM:
                models_to_run.append("LSM")
        else:
            model_map = {
                "ssm": ["SSM"],
                "attention": ["Attention"],
                "stack_rnn": ["StackRNN"] if HAS_STACK_RNN else [],
                "lsm": ["LSM"] if HAS_LSM else []
            }
            models_to_run = model_map.get(args.model, ["SSM", "Attention"])
            if not models_to_run:
                print(f"Requested model {args.model} is not available/implemented.")
                return 1

        # We will collect evaluation results across seeds and depths
        metrics_by_depth = {d: {m: {"token": [], "seq": []} for m in models_to_run} for d in eval_depths}

        for seed in seeds:
            print(f"\n--- Running Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # Generate training dataset: fixed depth=4, length=20, num_bracket_types=1
            X_train, Y_train = generate_nested_brackets(num_samples=num_samples, length=20, depth=4, num_bracket_types=1)

            # Instantiate models dynamically
            models = {}
            if "SSM" in models_to_run:
                models["SSM"] = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
            if "Attention" in models_to_run:
                models["Attention"] = CausalAttentionModel(vocab_size=2, d_model=8, state_dim=16)
            if "StackRNN" in models_to_run:
                models["StackRNN"] = StackRNN(vocab_size=2, hidden_size=16, stack_width=4, stack_depth=55)
            if "LSM" in models_to_run:
                models["LSM"] = LiquidStateMachine(input_size=2, reservoir_size=50, output_size=2, spectral_radius=0.99, sparsity=0.1)

            # Train models
            for m_name, model in models.items():
                print(f"Training {m_name} on nesting task...")
                train_model(model, X_train, Y_train, epochs=epochs, lr=0.03)

            # Evaluate across all testing depths
            for d in eval_depths:
                X_test, Y_test = generate_nested_brackets(num_samples=num_samples, length=2 * d, depth=d, num_bracket_types=1)

                for m_name, model in models.items():
                    token_acc, seq_acc = evaluate_model_accs(model, X_test, Y_test)
                    metrics_by_depth[d][m_name]["token"].append(token_acc)
                    metrics_by_depth[d][m_name]["seq"].append(seq_acc)

        # Compute mean metrics and write results/nesting_depth_results.csv
        csv_path = os.path.join(args.output_dir, "nesting_depth_results.csv")
        with open(csv_path, "w") as f:
            f.write("depth,model,token_accuracy,sequence_accuracy\n")
            for d in eval_depths:
                for model in models_to_run:
                    mean_token = np.mean(metrics_by_depth[d][model]["token"])
                    mean_seq = np.mean(metrics_by_depth[d][model]["seq"])
                    f.write(f"{d},{model},{mean_token:.4f},{mean_seq:.4f}\n")
        print(f"Detailed nesting depth results saved to {csv_path}")

        # Write results_table.csv as well (mean across all depths to maintain format)
        table_path = os.path.join(args.output_dir, "results_table.csv")
        with open(table_path, "w") as f:
            f.write("model,accuracy,token_accuracy,sequence_accuracy,sparsity\n")
            for model in models_to_run:
                all_token = []
                all_seq = []
                for d in eval_depths:
                    all_token.extend(metrics_by_depth[d][model]["token"])
                    all_seq.extend(metrics_by_depth[d][model]["seq"])
                mean_token = np.mean(all_token)
                mean_seq = np.mean(all_seq)
                f.write(f"{model},{mean_token:.4f},{mean_token:.4f},{mean_seq:.4f},0.0000\n")
        print(f"Summary table saved to {table_path}")

        # Generate the plot showing Accuracy vs Nesting Depth using matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        colors = {"SSM": "blue", "Attention": "red", "StackRNN": "green", "LSM": "purple"}
        markers = {"SSM": "o", "Attention": "s", "StackRNN": "^", "LSM": "d"}

        for model in models_to_run:
            color = colors.get(model, "black")
            marker = markers.get(model, "x")
            tokens = [np.mean(metrics_by_depth[d][model]["token"]) for d in eval_depths]
            seqs = [np.mean(metrics_by_depth[d][model]["seq"]) for d in eval_depths]

            plt.plot(eval_depths, seqs, label=f"{model} Sequence Accuracy", color=color, marker=marker, linestyle="-")
            plt.plot(eval_depths, tokens, label=f"{model} Token Accuracy", color=color, marker=marker, linestyle="--", alpha=0.5)

        plt.xlabel("Nesting Depth")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy vs. Nesting Depth")
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(args.output_dir, "accuracy_vs_depth.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Accuracy vs depth plot saved to {plot_path}")

        return 0
    
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
    if args.model == "all":
        models_to_run = ["SSM", "Attention", "Conv1D", "MarkovChain"]
        if HAS_LSM:
            models_to_run.append("LSM")
    else:
        model_map = {
            "ssm": ["SSM"],
            "attention": ["Attention"],
            "conv1d": ["Conv1D"],
            "markov": ["MarkovChain"],
            "stack_rnn": ["StackRNN"] if HAS_STACK_RNN else [],
            "lsm": ["LSM"] if HAS_LSM else []
        }
        models_to_run = model_map.get(args.model, ["SSM", "Attention", "Conv1D", "MarkovChain"])

    results = {m: {"token_accs": [], "seq_accs": [], "sparsities": []} for m in models_to_run}

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Generate train/val datasets
        X_train, Y_train = generate_alternating(train_len, num_samples)
        X_test, Y_test = generate_alternating(test_len, num_samples)

        # Instantiate models
        models = {}
        if "SSM" in models_to_run:
            models["SSM"] = RecurrentSSM(vocab_size=2, d_model=8, state_dim=16)
        if "Attention" in models_to_run:
            models["Attention"] = CausalAttentionModel(vocab_size=2, d_model=8, state_dim=16)
        if "Conv1D" in models_to_run:
            models["Conv1D"] = Conv1DModel(vocab_size=2, d_model=8, state_dim=16)
        if "MarkovChain" in models_to_run:
            models["MarkovChain"] = MarkovChainModel(vocab_size=2, d_model=8, state_dim=16)
        if "StackRNN" in models_to_run:
            models["StackRNN"] = StackRNN(vocab_size=2, hidden_size=16, stack_width=4, stack_depth=55)
        if "LSM" in models_to_run:
            models["LSM"] = LiquidStateMachine(input_size=2, reservoir_size=50, output_size=2, spectral_radius=0.99, sparsity=0.1)

        # Train models
        for m_name, model in models.items():
            print(f"Training {m_name}...")
            train_model(model, X_train, Y_train, epochs=epochs, lr=0.03)

        # Apply L0 pruning to SSM if present
        if "SSM" in models:
            print("Applying L0 pruning to SSM...")
            masked_ssm = apply_l0_mask(models["SSM"])
            l0_pruning_step(masked_ssm, temperature=0.1, target_sparsity=0.2)
            eval_ssm = masked_ssm
            ssm_sparsity = masked_ssm.pareto_frontier[-1].sparsity
        else:
            ssm_sparsity = 0.0

        # Evaluate models on test length (length generalization)
        for m_name, model in models.items():
            if m_name == "SSM":
                token_acc, seq_acc = evaluate_model_accs(eval_ssm, X_test, Y_test)
                results["SSM"]["token_accs"].append(token_acc)
                results["SSM"]["seq_accs"].append(seq_acc)
                results["SSM"]["sparsities"].append(ssm_sparsity)
            else:
                token_acc, seq_acc = evaluate_model_accs(model, X_test, Y_test)
                results[m_name]["token_accs"].append(token_acc)
                results[m_name]["seq_accs"].append(seq_acc)
                results[m_name]["sparsities"].append(0.0)

    # 3. Print stats and run statistical significance t-test / Mann-Whitney
    print("\n================ FINAL RESULTS ================")
    for name in models_to_run:
        mean_token = np.mean(results[name]["token_accs"])
        mean_seq = np.mean(results[name]["seq_accs"])
        mean_sparsity = np.mean(results[name]["sparsities"])
        print(f"{name}:")
        print(f"  Token Acc: {mean_token:.4f}")
        print(f"  Seq Acc:   {mean_seq:.4f}")
        print(f"  Sparsity:  {mean_sparsity:.4f}")

    if "SSM" in models_to_run and len(models_to_run) > 1:
        print("\n--- Statistical Significance (SSM vs Baselines) ---")
        for name in models_to_run:
            if name == "SSM":
                continue
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
        for name in models_to_run:
            mean_token = np.mean(results[name]["token_accs"])
            mean_seq = np.mean(results[name]["seq_accs"])
            mean_sparsity = np.mean(results[name]["sparsities"])
            f.write(f"{name},{mean_token:.4f},{mean_token:.4f},{mean_seq:.4f},{mean_sparsity:.4f}\n")
        
    print(f"Results saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
