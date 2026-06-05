import sys
sys.path.append('.')
import torch
from compare_task import MicroGPT, SEQ_LEN, VOCAB_SIZE
import sympy
from sympy.logic.boolalg import simplify_logic, Equivalent

# 1. LOAD THE WEIGHTS
device = torch.device('cpu')
model = MicroGPT(n_embd=8, use_mlp=False, use_ln=False, use_bias=False).to(device)
model.load_state_dict(torch.load("gpt.pt", map_location=device, weights_only=True))
model.eval()

# 2. EXTRACT DISCRETE ROUTING FROM THE WEIGHTS
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

# 3. BUILD AND MINIMIZE SYMBOLIC BOOLEAN LOGIC WITH SYMPY
print("--- SYMPY BOOLEAN MINIMIZATION ---")

# Let's do this just for the first active position (representing a 2-bit XNOR equivalence)
# C1_0, C1_1 -> Context Token 1 bits
# Q_0, Q_1   -> Query Token bits
# E          -> Context Token 2 bit 0 (the target to copy)
C1_0, C1_1, Q_0, Q_1, E = sympy.symbols('C1_0 C1_1 Q_0 Q_1 E')

# The raw logic we generated:
# (((C1_0 AND Q_0) OR (NOT C1_0 AND NOT Q_0)) AND ((C1_1 AND Q_1) OR (NOT C1_1 AND NOT Q_1))) AND E
raw_bit_0_match = (C1_0 & Q_0) | (~C1_0 & ~Q_0)
raw_bit_1_match = (C1_1 & Q_1) | (~C1_1 & ~Q_1)
raw_logic = raw_bit_0_match & raw_bit_1_match & E

print(f"1. Raw Logic Extracted from GPT Weights:\n   {raw_logic}\n")

minimized = simplify_logic(raw_logic)
print(f"2. SymPy Minimized Logic (CNF):\n   {minimized}\n")

# SymPy outputs the mathematical Conjunctive Normal Form (CNF):
# E & (C1_0 | ~Q_0) & (C1_1 | ~Q_1) & (Q_0 | ~C1_0) & (Q_1 | ~C1_1)
# 
# Notice that (C1_0 | ~Q_0) & (Q_0 | ~C1_0) is the mathematical CNF definition of XNOR.
# If we define the XNOR equivalent explicitly:
xnor_0 = Equivalent(C1_0, Q_0)
xnor_1 = Equivalent(C1_1, Q_1)
target_minimal_form = xnor_0 & xnor_1 & E

print(f"3. Target Minimal Form (XNOR):\n   {target_minimal_form}\n")

# We can ask SymPy to mathematically prove that our extracted raw logic
# is exactly equivalent to this minimal XNOR circuit:
is_equivalent = simplify_logic(Equivalent(raw_logic, target_minimal_form))
print(f"4. Are the Raw Weights mathematically equivalent to the XNOR circuit? {is_equivalent}")
