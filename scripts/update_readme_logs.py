import sys
import re

logs = sys.stdin.read()

# Filter out the \r step lines and keep only the final epoch reports and headers
clean_lines = []
for line in logs.split('\n'):
    if ("--- Training" in line) or ("--- Testing" in line) or ("Epoch" in line and "step" not in line) or ("| Test Acc" in line):
        clean_lines.append(line.strip())

# Extract MLP block
mlp_header = next(l for l in clean_lines if "--- Training 3-Layer MLP ---" in l)
mlp_epoch = next(l for l in clean_lines if "MLP - Epoch  40" in l)

# Extract GPT block
gpt_header = next(l for l in clean_lines if "--- Training MicroGPT" in l)
gpt_epoch = next(l for l in clean_lines if "GPT-PureAttn - Epoch  20" in l)

# Extract Boolean block
bool_header = next(l for l in clean_lines if "--- Testing Boolean Logic Circuit Equivalency ---" in l)
bool_epoch = next(l for l in clean_lines if "Boolean Circuit | Test Acc" in l)

# Extract LLVM block
llvm_header = next(l for l in clean_lines if "--- Testing LLVM-Optimized Integer Hardware Circuit ---" in l)
llvm_epoch = next(l for l in clean_lines if "LLVM Circuit | Test Acc" in l)

# Extract tests
bool_acc = re.search(r'Boolean Circuit \| Test Acc: ([\d.]+)%', logs)
llvm_acc = re.search(r'LLVM Circuit \| Test Acc: ([\d.]+)%', logs)
clj_acc = re.search(r'Clojure Circuit \| Test Acc: ([\d.]+)%', logs)
v_acc = re.search(r'Verilog Circuit \| Test Acc: ([\d.]+)%', logs)
apl_acc = re.search(r'APL Circuit \| Test Acc: ([\d.]+)%', logs)
mma_acc = re.search(r'Mathematica Circuit \| Test Acc: ([\d.]+)%', logs)
sympy_acc = re.search(r'SymPy Circuit \| Test Acc: ([\d.]+)%', logs)
shannon_acc = re.search(r'Shannon Markov \| Test Acc: ([\d.]+)%', logs)

bool_val = f"{bool_acc.group(1)}%" if bool_acc else "ERROR"
llvm_val = f"{llvm_acc.group(1)}%" if llvm_acc else "ERROR"
clj_val = f"{clj_acc.group(1)}%" if clj_acc else "ERROR"
v_val = f"{v_acc.group(1)}%" if v_acc else "ERROR"
apl_val = f"{apl_acc.group(1)}%" if apl_acc else "ERROR"
mma_val = f"{mma_acc.group(1)}%" if mma_acc else "ERROR"
sympy_val = f"{sympy_acc.group(1)}%" if sympy_acc else "ERROR"
shannon_val = f"{shannon_acc.group(1)}%" if shannon_acc else "ERROR"

# Extract Clojure block
clj_header = next(l for l in clean_lines if "--- Testing Clojure Integer Hardware Circuit ---" in l)
clj_epoch = next(l for l in clean_lines if "Clojure Circuit | Test Acc" in l)

# Extract Mathematica block
mma_header = next(l for l in clean_lines if "--- Testing Mathematica Hardware Circuit ---" in l)
mma_epoch = next(l for l in clean_lines if "Mathematica Circuit | Test Acc" in l)

# Extract APL block
apl_header = next(l for l in clean_lines if "--- Testing APL Array Language Circuit ---" in l)
apl_epoch = next(l for l in clean_lines if "APL Circuit | Test Acc" in l)

# Extract Verilog block
v_header = next(l for l in clean_lines if "--- Testing Verilog RTL Hardware Circuit ---" in l)
v_epoch = next(l for l in clean_lines if "Verilog Circuit | Test Acc" in l)

# Extract SymPy block
sympy_header = next(l for l in clean_lines if "--- Testing SymPy Functional Abstraction Circuit ---" in l)
sympy_epoch = next(l for l in clean_lines if "SymPy Circuit | Test Acc" in l)

# Extract Neurosymbolic block
neuro_header = next(l for l in clean_lines if "--- Training SymPy-Structured MLP Network ---" in l)
neuro_epoch = next(l for l in clean_lines if "SymPy-Structured MLP | Test Acc" in l)

# Extract Shannon block
shannon_header = next(l for l in clean_lines if "--- Testing Shannon N-Gram Markov Baseline ---" in l)
shannon_epoch = next(l for l in clean_lines if "Shannon Markov | Test Acc" in l)

new_block = f"""<!-- PERFORMANCE LOGS START -->
```text
{mlp_header}
{mlp_epoch}

{gpt_header}
{gpt_epoch}

{neuro_header}
{neuro_epoch}

{shannon_header}
{shannon_epoch}

{bool_header}
{bool_epoch}

{sympy_header}
{sympy_epoch}

{llvm_header}
{llvm_epoch}

{clj_header}
{clj_epoch}

{mma_header}
{mma_epoch}

{apl_header}
{apl_epoch}

{v_header}
{v_epoch}
```
<!-- PERFORMANCE LOGS END -->"""

with open("README.md", "r") as f:
    readme = f.read()

readme = re.sub(r'<!-- PERFORMANCE LOGS START -->.*?<!-- PERFORMANCE LOGS END -->', new_block, readme, flags=re.DOTALL)

with open("README.md", "w") as f:
    f.write(readme)

# Extract Shakespeare block
try:
    shake_start1 = logs.find("--- Training Pure Connectionist MLP-Transformer on Regular Expressions ---")
    shake_start2 = logs.find("--- Training Pure Connectionist Recurrent-MLP LM on Regular Expressions ---")
    shake_start3 = logs.find("--- Training Pure Connectionist Sparse Recurrent-MLP on Regular Expressions ---")
    shake_start4 = logs.find("--- Training Pure Connectionist Vanilla TDL on Regular Expressions ---")
    shake_start5 = logs.find("--- Training Pure Connectionist Vanilla TDL (1990s SGD) on Regular Expressions ---")
    shake_start6 = logs.find("--- Testing Shannon's 1948 Markovian Text Generator ---")
    
    if shake_start1 != -1 and shake_start2 != -1 and shake_start3 != -1 and shake_start4 != -1 and shake_start5 != -1 and shake_start6 != -1:
        shake_block1 = logs[shake_start1:shake_start2].strip()
        shake_block2 = logs[shake_start2:shake_start3].strip()
        shake_block3_full = logs[shake_start3:shake_start4].strip()
        shake_block4 = logs[shake_start4:shake_start5].strip()
        shake_block5 = logs[shake_start5:shake_start6].strip()
        
        shake_end6 = logs.find("Initializing Z3 SMT Solver", shake_start6)
        if shake_end6 == -1: shake_end6 = len(logs)
        shake_block6 = logs[shake_start6:shake_end6].strip()
        
        # Split the text logs from the mermaid graph
        if "--- Physical Neural Topology (Mermaid) ---" in shake_block3_full:
            shake_block3_text, mermaid_block = shake_block3_full.split("--- Physical Neural Topology (Mermaid) ---")
            shake_block3_text = shake_block3_text.strip()
            mermaid_block = mermaid_block.strip()
            
            new_shake = f"<!-- REGEX LOGS START -->\n```text\n{shake_block1}\n```\n\n```text\n{shake_block2}\n```\n\n```text\n{shake_block3_text}\n```\n\n```text\n{shake_block4}\n```\n\n```text\n{shake_block5}\n```\n\n```text\n{shake_block6}\n```\n<!-- REGEX LOGS END -->"
            
            brain_injection = f"<!-- LITERAL TRAINED BRAIN START -->\n#### The Literal Trained Brain\n\n{mermaid_block}\n<!-- LITERAL TRAINED BRAIN END -->"
            readme = re.sub(r'<!-- LITERAL TRAINED BRAIN START -->.*?<!-- LITERAL TRAINED BRAIN END -->', lambda _: brain_injection, readme, flags=re.DOTALL)
        else:
            new_shake = f"<!-- REGEX LOGS START -->\n```text\n{shake_block1}\n```\n\n```text\n{shake_block2}\n```\n\n```text\n{shake_block3_full}\n```\n\n```text\n{shake_block4}\n```\n\n```text\n{shake_block5}\n```\n<!-- REGEX LOGS END -->"
        readme = re.sub(r'<!-- REGEX LOGS START -->.*?<!-- REGEX LOGS END -->', lambda _: new_shake, readme, flags=re.DOTALL)
        with open("README.md", "w") as f:
            f.write(readme)
except Exception as e:
    print(f"Error parsing shakespeare logs: {e}")

# Extract Z3 block
try:
    z3_start = logs.find("Initializing Z3 SMT Solver")
    if z3_start != -1:
        z3_end = logs.find("This mathematical proof confirms", z3_start)
        z3_block = logs[z3_start:z3_end + 80].strip()
        z3_injection = f"""<!-- Z3 SUPEROPTIMIZATION START -->\n```text\n{z3_block}\n```\n<!-- Z3 SUPEROPTIMIZATION END -->"""
        readme = re.sub(r'<!-- Z3 SUPEROPTIMIZATION START -->.*?<!-- Z3 SUPEROPTIMIZATION END -->', lambda _: z3_injection, readme, flags=re.DOTALL)
        with open("README.md", "w") as f:
            f.write(readme)
except Exception as e:
    print(f"Error parsing z3 logs: {e}")

print("README.md has been automatically updated with the latest performance logs!")
