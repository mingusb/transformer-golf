import sys
sys.path.append('.')
import torch
from compare_task import MicroGPT, SEQ_LEN, VOCAB_SIZE
import sympy
from sympy.logic.boolalg import simplify_logic, Equivalent
import math
import re

# 1. LOAD THE WEIGHTS & GET PARAMS
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

# 2. GENERATE AND MINIMIZE SYMBOLIC BOOLEAN LOGIC
ctx_vars = [[sympy.Symbol(f'C{j}_{i}') for i in range(num_bits)] for j in range(context_len)]
q_vars = [sympy.Symbol(f'Q_{i}') for i in range(num_bits)]

y_0_terms = []
for j in active_positions:
    bit_matches = []
    for i in range(num_bits):
        c_bit = ctx_vars[j][i]
        q_bit = q_vars[i]
        bit_matches.append((c_bit & q_bit) | (~c_bit & ~q_bit))
    
    full_match = bit_matches[0]
    for m in bit_matches[1:]:
        full_match = full_match & m
    y_0_terms.append(full_match & ctx_vars[j+1][0])

raw_y_0 = y_0_terms[0]
for term in y_0_terms[1:]:
    raw_y_0 = raw_y_0 | term

simplified = simplify_logic(raw_y_0)

# 3. AUTOMATIC FUNCTIONAL ABSTRACTION
print(f"--- AUTOMATIC FUNCTIONAL ABSTRACTION ({num_bits}-bit) ---\n")

# SymPy's output is an 'Or' chain of repeated identical blocks
# We automatically extract the structural signature of the very first block
if isinstance(simplified, sympy.Or):
    first_block = simplified.args[0]
else:
    first_block = simplified

# Map the pairs of bits (C0_i and Q_i) to generic letters (A,B, C,D, E,F, G,H)
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
subs_map = {}

# Map the target bit (C1_0) to Z
target_bit = ctx_vars[active_positions[0] + 1][0]
target_letter = sympy.Symbol('Z')
subs_map[target_bit] = target_letter

for i in range(num_bits):
    ctx_bit = ctx_vars[active_positions[0]][i]
    qry_bit = q_vars[i]
    
    l1 = sympy.Symbol(letters[i*2])
    l2 = sympy.Symbol(letters[i*2 + 1])
        
    subs_map[ctx_bit] = l1
    subs_map[qry_bit] = l2

# Perform the symbolic abstraction
abstract_function = first_block.subs(subs_map)

# Format the output visually to use the word XNOR
abstract_str = str(abstract_function)

# 4. Generate Valid Executable Python Code
print(f"2. Rolled up into Executable Abstract Python Function:\n")
print(f"   def XNOR(x, y):")
print(f"       return ~(x ^ y) & 1\n")
print(f"   def induction_match(Context_Token, Query_Token, Z):")
print(f"       # Unpack the bits from the integer tokens")

for i in range(num_bits):
    print(f"       {letters[i*2]} = (Context_Token >> {i}) & 1")
    print(f"       {letters[i*2+1]} = (Query_Token >> {i}) & 1")

# Regex to dynamically match any letter pair ((A & B) | (~A & ~B)) and replace with XNOR(A, B)
py_abstract_str = str(abstract_function)
py_abstract_str = re.sub(r'\(\(([A-Z]) & ([A-Z])\) \| \(~\1 & ~\2\)\)', r'XNOR(\1, \2)', py_abstract_str)

# Use valid python bitwise '&' instead of 'AND'
py_abstract_str = py_abstract_str.replace(' & ', ' & ').replace(' | ', ' | ')

print(f"\n       return {py_abstract_str}")

print("\n3. Final Unrolled Sequence Execution (Map-Reduce):")
print(f"   y = 0")
print(f"   for j in range({len(active_positions)}):")
print(f"       # Extract the target bit Z from the next token")
print(f"       Z = (Context[j+1] >> 0) & 1")
print(f"       y |= induction_match(Context[j], Query, Z)")
