import torch
import torch.nn as nn
import numpy as np

def not_gate(x):
    if isinstance(x, torch.Tensor):
        return 1.0 - x
    return 1.0 - x

def and_gate(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple, torch.Tensor)):
        inputs = args[0]
    else:
        inputs = args
    if isinstance(inputs, torch.Tensor):
        n = inputs.shape[-1]
        return torch.relu(inputs.sum(dim=-1) - (n - 1))
    else:
        n = len(inputs)
        sum_val = sum(inputs)
        if isinstance(sum_val, torch.Tensor):
            return torch.relu(sum_val - (n - 1))
        return max(0.0, sum_val - (n - 1))

def or_gate(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple, torch.Tensor)):
        inputs = args[0]
    else:
        inputs = args
    if isinstance(inputs, torch.Tensor):
        return torch.clamp(inputs.sum(dim=-1), 0.0, 1.0)
    else:
        sum_val = sum(inputs)
        if isinstance(sum_val, torch.Tensor):
            return torch.clamp(sum_val, 0.0, 1.0)
        return min(1.0, max(0.0, sum_val))

class GateModule(nn.Module):
    def __init__(self, op, arg_indices, num_inputs):
        super().__init__()
        self.op = op
        self.arg_indices = arg_indices
        self.linear = nn.Linear(num_inputs, 1, bias=True)
        
        # Set exact weights and bias
        with torch.no_grad():
            weight = torch.zeros(1, num_inputs)
            if op == "NOT":
                weight[0, arg_indices[0]] = -1.0
            else:
                for idx in arg_indices:
                    weight[0, idx] += 1.0
            self.linear.weight.copy_(weight)
            
            if op == "AND":
                n = len(arg_indices)
                bias = torch.tensor([float(-(n - 1))])
            elif op == "NOT":
                bias = torch.tensor([1.0])
            else:
                bias = torch.tensor([0.0])
            self.linear.bias.copy_(bias)
            
    def forward(self, V):
        # V shape: (batch, alphabet_size, num_inputs)
        out = self.linear(V) # shape: (batch, alphabet_size, 1)
        if self.op in ("AND", "NOT", "BUF", "IDENTITY", "BUFFER"):
            out = torch.relu(out)
        elif self.op == "OR":
            out = torch.clamp(out, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown operator {self.op}")
        return out

class MapReduceMLP(nn.Module):
    def __init__(self, circuit, alphabet_size):
        super().__init__()
        if not circuit:
            raise ValueError("Circuit cannot be empty")
        self.circuit = circuit
        self.alphabet_size = alphabet_size
        
        inputs = circuit.get("inputs", [])
        if len(inputs) != len(set(inputs)):
            raise ValueError("Duplicate input names are not allowed in the circuit")
        self.inputs = inputs
        
        # Sort gates topologically
        gates = circuit.get("gates", {})
        resolved = set(inputs)
        pending = list(gates.keys())
        sorted_gates = []
        
        max_iters = len(gates) * len(gates) + 10
        iters = 0
        while pending and iters < max_iters:
            next_pending = []
            for g in pending:
                gate = gates[g]
                if isinstance(gate, tuple):
                    op, args = gate
                else:
                    op = gate.get("type", gate.get("op"))
                    args = gate.get("inputs", gate.get("args", []))
                if all(arg in resolved for arg in args):
                    sorted_gates.append((g, op, args))
                    resolved.add(g)
                else:
                    next_pending.append(g)
            pending = next_pending
            iters += 1
            
        if pending:
            raise ValueError("Circuit contains unresolved gate dependencies or cycles")
            
        # Build gates modules and mapping
        self.vars_map = {name: i for i, name in enumerate(inputs)}
        self.gates_modules = nn.ModuleList()
        for g, op, args in sorted_gates:
            arg_indices = [self.vars_map[arg] for arg in args]
            num_inputs = len(self.vars_map)
            module = GateModule(op, arg_indices, num_inputs)
            self.gates_modules.append(module)
            self.vars_map[g] = len(self.vars_map)

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a PyTorch Tensor")
        
        if X.dim() == 2:
            X = torch.nn.functional.one_hot(X.long(), num_classes=self.alphabet_size).float()
            
        batch, seq_len, alphabet_size = X.shape
        if torch.isinf(X).any() or torch.isnan(X).any():
            X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        if batch == 0:
            return torch.zeros(0, seq_len, alphabet_size, device=X.device)

        # Prepare V tensor: shape (batch, alphabet_size, seq_len)
        V = X.permute(0, 2, 1)
        
        expected_seq_len = len(self.inputs)
        if seq_len < expected_seq_len:
            # Pad with zeros
            pad_size = expected_seq_len - seq_len
            padding = torch.zeros(batch, alphabet_size, pad_size, device=X.device)
            V = torch.cat([V, padding], dim=-1)
        elif seq_len > expected_seq_len:
            V = V[:, :, :expected_seq_len]
            
        # Run compiled gates
        for module in self.gates_modules:
            out = module(V)
            V = torch.cat([V, out], dim=-1)
            
        # Extract outputs
        outputs = self.circuit.get("outputs", [])
        out_tensors = []
        for j in range(seq_len):
            target_var = None
            if isinstance(outputs, dict):
                key = f"y_{j}"
                if key in outputs:
                    target_var = outputs[key]
                elif f"g_{j}" in self.vars_map:
                    target_var = f"g_{j}"
            elif isinstance(outputs, list):
                if j < len(outputs):
                    target_var = outputs[j]
                elif f"g_{j}" in self.vars_map:
                    target_var = f"g_{j}"
            else:
                if f"g_{j}" in self.vars_map:
                    target_var = f"g_{j}"
            
            if target_var is not None and target_var in self.vars_map:
                idx = self.vars_map[target_var]
                out_tensors.append(V[:, :, idx])
            else:
                out_tensors.append(torch.zeros(batch, alphabet_size, device=X.device))
                
        if len(out_tensors) == 0:
            return torch.zeros(batch, 0, alphabet_size, device=X.device)
        return torch.stack(out_tensors, dim=1)

def compile_mlp_from_circuit(circuit, alphabet_size):
    return MapReduceMLP(circuit, alphabet_size)
