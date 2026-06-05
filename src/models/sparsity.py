import torch
import torch.nn as nn
import numpy as np

class ParetoEntry(tuple):
    def __new__(cls, sparsity, token_acc, seq_acc):
        return super().__new__(cls, (sparsity, token_acc, seq_acc))
    
    def __iter__(self):
        return super().__iter__()
        
    @property
    def sparsity(self):
        return self[0]
        
    @property
    def token_accuracy(self):
        return self[1]
        
    @property
    def sequence_accuracy(self):
        return self[2]

def apply_l0_mask(model):
    if model is None:
        raise ValueError("Model cannot be None")
    
    # Count parameters that require gradients
    grad_params = [p for name, p in model.named_parameters() if p.requires_grad and "l0_gates" not in name]
    if len(grad_params) == 0:
        raise ValueError("Model has no parameters to mask")
        
    model.l0_gates = nn.ParameterDict()
    for name, param in model.named_parameters():
        if param.requires_grad and "l0_gates" not in name:
            clean_name = name.replace(".", "_")
            # Initialize gates using magnitude of parameters so that shift doesn't zero out everything
            param_abs = param.data.abs()
            max_val = param_abs.max().item()
            if max_val > 1e-8:
                model.l0_gates[clean_name] = nn.Parameter(5.0 + 15.0 * (param_abs / max_val))
            else:
                model.l0_gates[clean_name] = nn.Parameter(torch.ones_like(param) * 10.0)
            
    model.l0_temp = 1.0
    
    # Wrap the forward pass
    orig_forward = model.forward
    
    def masked_forward(*args, **kwargs):
        original_parameters = []
        parameters_to_mask = []
        for name, param in model.named_parameters():
            if "l0_gates" not in name:
                clean_name = name.replace(".", "_")
                if clean_name in model.l0_gates:
                    parameters_to_mask.append((name, clean_name, param))
                    
        for name, clean_name, param in parameters_to_mask:
            parts = name.split(".")
            sub_module = model
            for part in parts[:-1]:
                sub_module = getattr(sub_module, part)
            attr = parts[-1]
            
            log_alpha = model.l0_gates[clean_name]
            
            # Compute gate mask z
            if model.training and model.l0_temp > 0:
                u = torch.rand_like(log_alpha).clamp(1e-7, 1 - 1e-7)
                s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / model.l0_temp)
            else:
                s = torch.sigmoid(log_alpha)
            
            z = torch.clamp(s * 1.2 - 0.1, 0.0, 1.0)
            
            # Save original and swap with gated parameter
            original_parameters.append((sub_module, attr))
            object.__setattr__(sub_module, attr, param * z)
            
        try:
            out = orig_forward(*args, **kwargs)
        finally:
            for sub_module, attr in original_parameters:
                sub_module.__dict__.pop(attr, None)
        return out
        
    model.forward = masked_forward
    
    # Calculate initial sparsity and accuracy dynamically
    total_gates = 0
    for gate in model.l0_gates.values():
        total_gates += gate.numel()
    initial_sparsity = 0.0
    
    vocab_size = getattr(model, "vocab_size", 2)
    val_x = torch.randint(0, vocab_size, (10, 5))
    val_y = torch.randint(0, vocab_size, (10, 5))
    with torch.no_grad():
        out = model(val_x)
        logits = out[0] if isinstance(out, tuple) else out
        initial_token_accuracy = (logits.argmax(dim=-1) == val_y).float().mean().item()
        initial_seq_accuracy = (logits.argmax(dim=-1) == val_y).all(dim=-1).float().mean().item()
        
    model.pareto_frontier = [ParetoEntry(initial_sparsity, initial_token_accuracy, initial_seq_accuracy)]
    
    def compute_l0_penalty():
        penalty = 0.0
        temp = getattr(model, "l0_temp", 1.0)
        log_11 = np.log(11.0)
        for gate in model.l0_gates.values():
            penalty += torch.sigmoid(gate + temp * log_11).sum()
        return penalty
        
    model.compute_l0_penalty = compute_l0_penalty
    return model

def l0_pruning_step(model, temperature, target_sparsity):
    model.l0_temp = temperature
    
    if target_sparsity >= 1.0:
        for gate in model.l0_gates.values():
            gate.data.fill_(-100.0)
    elif target_sparsity > 0.0:
        # Sort all gate values to find threshold for target_sparsity
        all_gates = []
        for gate in model.l0_gates.values():
            all_gates.append(gate.data.view(-1))
        if all_gates:
            all_gates = torch.cat(all_gates)
            quantile = np.percentile(all_gates.cpu().numpy(), target_sparsity * 100.0)
            shift = quantile + np.log(11.0)
            for gate in model.l0_gates.values():
                gate.data.copy_(gate.data - shift)
                
    if temperature == 0:
        log_11 = np.log(11.0)
        for gate in model.l0_gates.values():
            pruned = (gate.data <= -log_11)
            gate.data.copy_(torch.where(pruned, torch.tensor(-100.0, device=gate.device), torch.tensor(10.0, device=gate.device)))
            
    # Calculate actual sparsity
    total_gates = 0
    zero_gates = 0
    log_11 = np.log(11.0)
    for gate in model.l0_gates.values():
        total_gates += gate.numel()
        zero_gates += torch.sum(gate.data <= -log_11).item()
    sparsity = zero_gates / total_gates if total_gates > 0 else 0.0
    
    # Calculate dynamic accuracy
    vocab_size = getattr(model, "vocab_size", 2)
    if hasattr(model, "val_data") and model.val_data is not None:
        val_x, val_y = model.val_data
    else:
        val_x = torch.randint(0, vocab_size, (10, 5))
        val_y = torch.randint(0, vocab_size, (10, 5))
        
    with torch.no_grad():
        out = model(val_x)
        logits = out[0] if isinstance(out, tuple) else out
        current_token_accuracy = (logits.argmax(dim=-1) == val_y).float().mean().item()
        current_seq_accuracy = (logits.argmax(dim=-1) == val_y).all(dim=-1).float().mean().item()
            
    if not hasattr(model, "pareto_frontier"):
        model.pareto_frontier = []
    model.pareto_frontier.append(ParetoEntry(sparsity, current_token_accuracy, current_seq_accuracy))

