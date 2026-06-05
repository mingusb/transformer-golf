def test_llvm():
    import json
    import ctypes
    import os
    import numpy as np
    
    with open("docs/test_data.json", "r") as f:
        data = json.load(f)
        
    _lib = ctypes.CDLL(os.path.abspath("optimized_true_gpt.so"))
    _lib.predict_next_token_optimized.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64]
    _lib.predict_next_token_optimized.restype = ctypes.c_int64
    
    correct = 0
    for i in range(len(data['x'])):
        ctx_arr = np.array(data['x'][i], dtype=np.int64)
        ctx_ptr = ctx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        if _lib.predict_next_token_optimized(ctx_ptr, len(data['x'][i]), data['q'][i]) == data['y'][i]:
            correct += 1
            
    print("\\n--- Testing LLVM-Optimized Integer Hardware Circuit ---")
    print(f"LLVM Circuit | Test Acc: {(correct * 100.0) / len(data['x']):.1f}%")

import json
import os
import subprocess

def test_clojure():
    # Load JSON and write as Clojure file
    with open("docs/test_data.json", "r") as f:
        data = json.load(f)
    
    with open("docs/test_data.clj", "w") as f:
        # Python lists with commas are valid Clojure vectors!
        f.write(f"(def test-x {data['x']})\n")
        f.write(f"(def test-q {data['q']})\n")
        f.write(f"(def test-y {data['y']})\n")
        
    clj_runner = """
(load-file "optimized_true_gpt.clj")
(load-file "docs/test_data.clj")

(def results (map (fn [ctx q tgt] 
                    (if (= (predict-next-token-optimized ctx q) tgt) 1 0))
                  test-x test-q test-y))

(println (format "%.1f" (float (* 100 (/ (reduce + results) (count results))))))
"""
    with open("test_runner.clj", "w") as f:
        f.write(clj_runner)
        
    print("\n--- Testing Clojure Integer Hardware Circuit ---")
    result = subprocess.run(["clojure", "-M", "test_runner.clj"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Clojure Circuit | Test Acc: ERROR")
        print(result.stderr)
    else:
        acc = result.stdout.strip()
        print(f"Clojure Circuit | Test Acc: {acc}%")

def test_verilog():
    with open("docs/test_data.json", "r") as f:
        data = json.load(f)
        
    seq_len = len(data['x'][0])
    num_bits = 4
    
    # Write a Verilog testbench
    tb = []
    tb.append("`timescale 1ns/1ps")
    tb.append("module tb;")
    tb.append(f"    reg [{seq_len*num_bits}-1:0] context_flat;")
    tb.append(f"    reg [{num_bits}-1:0] query;")
    tb.append(f"    wire [{num_bits}-1:0] y;")
    tb.append(f"    integer correct;")
    tb.append(f"    integer i;")
    tb.append("")
    tb.append(f"    predict_next_token_optimized #(.NUM_BITS({num_bits})) uut (")
    # Dynamically connect the unpacked ports
    for j in range(seq_len):
        tb.append(f"        .context_{j}(context_flat[{j*num_bits + num_bits - 1}:{j*num_bits}]),")
    tb.append(f"        .query(query),")
    tb.append(f"        .y(y)")
    tb.append(f"    );")
    tb.append("")
    tb.append("    initial begin")
    tb.append("        correct = 0;")
    
    # Generate the stimulus for all 1000 items
    for i in range(len(data['x'])):
        ctx = data['x'][i]
        q = data['q'][i]
        tgt = data['y'][i]
        
        # Flatten the context array into a single integer for the Verilog reg
        flat_val = 0
        for j in range(seq_len):
            flat_val |= (ctx[j] << (j * num_bits))
            
        tb.append(f"        context_flat = {seq_len*num_bits}'d{flat_val}; query = {num_bits}'d{q}; #1;")
        tb.append(f"        if (y == {num_bits}'d{tgt}) correct = correct + 1;")
    
    tb.append(f"        $display(\"%.1f\", (correct * 100.0) / {len(data['x'])});")
    tb.append("        $finish;")
    tb.append("    end")
    tb.append("endmodule")
    
    with open("test_tb.v", "w") as f:
        f.write("\n".join(tb))
        
    print("\n--- Testing Verilog RTL Hardware Circuit ---")
    # Compile with iverilog
    subprocess.run(["iverilog", "-o", "sim.vvp", "optimized_true_gpt.v", "test_tb.v"], capture_output=True)
    # Execute with vvp
    result = subprocess.run(["vvp", "sim.vvp"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Verilog Circuit | Test Acc: ERROR")
        print(result.stderr)
    else:
        # Strip out the $finish output from Icarus Verilog
        lines = result.stdout.strip().split('\n')
        acc = lines[0] if lines else "0.0"
        print(f"Verilog Circuit | Test Acc: {acc}%")

def test_apl():
    with open("docs/test_data.json", "r") as f:
        data = json.load(f)
        
    with open("optimized_true_gpt.apl", "r") as f:
        apl_impl = f.read().strip()
        
    apl_script = []
    apl_script.append(apl_impl)
    apl_script.append("correct ← 0")
    
    for i in range(len(data['x'])):
        ctx_str = " ".join(map(str, data['x'][i]))
        apl_script.append(f"y ← ({ctx_str}) predict_next_token {data['q'][i]}")
        apl_script.append(f"correct ← correct + y = {data['y'][i]}")
        
    apl_script.append(f"⎕ ← (correct × 100) ÷ {len(data['x'])}")
    
    with open("test_runner.apl", "w") as f:
        f.write("\n".join(apl_script) + "\n")
        
    print("\n--- Testing APL Array Language Circuit ---")
    result = subprocess.run(["node", "ngn-apl/apl.js", "test_runner.apl"], capture_output=True, text=True)
    if result.returncode != 0:
        print("APL Circuit | Test Acc: ERROR")
        print(result.stderr)
    else:
        acc = result.stdout.strip()
        print(f"APL Circuit | Test Acc: {float(acc):.1f}%")

def test_mathematica():
    with open("docs/test_data.json", "r") as f:
        data = json.load(f)
        
    with open("optimized_true_gpt.wls", "r") as f:
        mma_impl = f.read().strip()
        
    mma_script = []
    mma_script.append(mma_impl)
    mma_script.append("correct = 0;")
    
    for i in range(len(data['x'])):
        ctx_str = "{" + ", ".join(map(str, data['x'][i])) + "}"
        mma_script.append(f"y = PredictNextToken[{ctx_str}, {data['q'][i]}];")
        mma_script.append(f"correct = correct + Boole[y == {data['y'][i]}];")
        
    mma_script.append(f"Print[N[(correct * 100) / {len(data['x'])}]];")
    
    with open("test_runner.wls", "w") as f:
        f.write("\n".join(mma_script) + "\n")
        
    print("\n--- Testing Mathematica Hardware Circuit ---")
    
    # We must pipe the script into mathics since `mathics test_runner.wls` sometimes triggers REPL
    with open("test_runner.wls", "r") as f:
        script_content = f.read()
    
    result = subprocess.run(["./venv/bin/mathics", "-c", script_content], capture_output=True, text=True)
    if result.returncode != 0:
        print("Mathematica Circuit | Test Acc: ERROR")
        print(result.stderr)
    else:
        # Mathics output might have warnings or blank lines. Find the last numerical line.
        lines = [l.strip() for l in result.stdout.split('\n') if l.strip()]
        # The result of Print[] will be in the output, e.g. "100."
        acc = "0.0"
        for line in reversed(lines):
            try:
                acc = str(float(line))
                break
            except:
                pass
        print(f"Mathematica Circuit | Test Acc: {float(acc):.1f}%")

def test_sympy():
    with open("docs/test_data.json", "r") as f:
        data = json.load(f)
        
    import sys
    sys.path.append('.')
    import optimized_true_gpt_sympy
    
    correct = 0
    for i in range(len(data['x'])):
        ctx = data['x'][i]
        q = data['q'][i]
        tgt = data['y'][i]
        if optimized_true_gpt_sympy.predict_next_token_sympy(ctx, q) == tgt:
            correct += 1
            
    print("\n--- Testing SymPy Functional Abstraction Circuit ---")
    print(f"SymPy Circuit | Test Acc: {(correct * 100.0) / len(data['x']):.1f}%")

if __name__ == "__main__":
    test_llvm()
    test_clojure()
    test_sympy()
    test_mathematica()
    test_apl()
    test_verilog()
