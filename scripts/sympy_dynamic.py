import sys
sys.path.append('.')
import torch
from compare_task import MicroGPT, SEQ_LEN, VOCAB_SIZE
import sympy
from sympy.logic.boolalg import simplify_logic, Equivalent
import math

# 1. LOAD THE WEIGHTS
device = torch.device('cpu')
model = MicroGPT(n_embd=8, use_mlp=False, use_ln=False, use_bias=False).to(device)
model.load_state_dict(torch.load("gpt.pt", map_location=device, weights_only=True))
model.eval()

context_len = SEQ_LEN - 1
active_positions = []
for pos in range(context_len - 1):
    test_seq = list(range(context_len))
    query = test_seq[pos]
    test_seq.append(query)
    x = torch.tensor([test_seq], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1).item()
    if pred == test_seq[pos + 1]:
        active_positions.append(pos)

num_bits = max(1, math.ceil(math.log2(VOCAB_SIZE)))

print(f"--- DYNAMIC AUTOMATIC SYMPY MINIMIZATION ({num_bits}-bit) ---")

# Dynamically generate SymPy variables for the entire sequence
ctx_vars = [[sympy.Symbol(f'C{j}_{i}') for i in range(num_bits)] for j in range(context_len)]
q_vars = [sympy.Symbol(f'Q_{i}') for i in range(num_bits)]

# We will minimize the logic for the 0th bit of the output (Y_0)
y_0_terms = []

for j in active_positions:
    # Match condition: ALL bits of Context[j] must equal Query
    bit_matches = []
    for i in range(num_bits):
        c_bit = ctx_vars[j][i]
        q_bit = q_vars[i]
        # Raw unoptimized XNOR gate from PyTorch extraction
        match = (c_bit & q_bit) | (~c_bit & ~q_bit)
        bit_matches.append(match)
    
    # AND gate across all bit matches
    full_match = bit_matches[0]
    for m in bit_matches[1:]:
        full_match = full_match & m
        
    # If match, copy Context[j+1] bit 0
    target_bit = ctx_vars[j+1][0]
    y_0_terms.append(full_match & target_bit)

# OR gate across all possible active positions
raw_y_0 = y_0_terms[0]
for term in y_0_terms[1:]:
    raw_y_0 = raw_y_0 | term

print(f"1. Dynamically Generated Raw Logic for Output Bit 0 (Length: {len(str(raw_y_0))} chars)")
print("2. Running SymPy simplification algorithm (this may take a moment depending on num_bits)...")

simplified = simplify_logic(raw_y_0)

simplified_str = str(simplified)
import re
# SymPy mathematically expands XNOR to its CNF form: ((C & Q) | (~C & ~Q)).
# We will use regex to visually fold this back into the literal word 'XNOR' for clarity:
pretty_xnor = re.sub(r'\(\((C\d+_\d+) & (Q_\d+)\) \| \(~\1 & ~\2\)\)', r'(\1 XNOR \2)', simplified_str)
# Replace & with AND, | with OR
pretty_xnor = pretty_xnor.replace(' & ', ' AND ').replace(' | ', ' OR\n')

print(f"3. Automatically Minimized Form (Formatted):\n{pretty_xnor}")
