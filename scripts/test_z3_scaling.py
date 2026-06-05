import z3
import time

def test_scaling(bits):
    print(f"\n--- Testing synthesis for {bits}-bit match ---")
    start = time.time()
    
    # Inputs: C_0..C_N-1, Q_0..Q_N-1, E
    num_vars = 2 * bits + 1
    
    # Target func
    def eval_target(state):
        c = [(state >> i) & 1 for i in range(bits)]
        q = [(state >> (bits + i)) & 1 for i in range(bits)]
        e = (state >> (2 * bits)) & 1
        
        match = True
        for i in range(bits):
            if c[i] != q[i]:
                match = False
                break
        return 1 if (match and e) else 0

    print(f"Truth table size: {2**num_vars} rows")
    
    # Expected gate count: 1 XOR per bit, plus AND tree.
    # For bits=2, gates=4. For bits=3, gates=6. For bits=4, gates=8.
    N_gates = 2 * bits
    
    solver = z3.Solver()
    
    op = [z3.Int(f'op_{i}') for i in range(num_vars, num_vars+N_gates)]
    in_A = [z3.Int(f'inA_{i}') for i in range(num_vars, num_vars+N_gates)]
    in_B = [z3.Int(f'inB_{i}') for i in range(num_vars, num_vars+N_gates)]
    
    for i in range(N_gates):
        idx = num_vars + i
        solver.add(op[i] >= 0, op[i] <= 15)
        solver.add(in_A[i] >= 0, in_A[i] < idx)
        solver.add(in_B[i] >= 0, in_B[i] < idx)
        
    for state in range(2**num_vars):
        target_val = eval_target(state)
        v = [z3.Bool(f'v_{idx}_{state}') for idx in range(num_vars+N_gates)]
        
        for i in range(bits):
            solver.add(v[i] == bool((state >> i) & 1))
            solver.add(v[bits + i] == bool((state >> (bits + i)) & 1))
        solver.add(v[2 * bits] == bool((state >> (2 * bits)) & 1))
        
        for i in range(N_gates):
            idx = num_vars + i
            
            def select_val(src_var):
                res = v[0]
                for j in range(1, idx):
                    res = z3.If(src_var == j, v[j], res)
                return res
            
            val_A = select_val(in_A[i])
            val_B = select_val(in_B[i])
            
            res = z3.BoolVal(False)
            for op_code in range(16):
                bit0 = (op_code >> 0) & 1
                bit1 = (op_code >> 1) & 1
                bit2 = (op_code >> 2) & 1
                bit3 = (op_code >> 3) & 1
                op_expr = z3.If(val_A, z3.If(val_B, bit3 == 1, bit2 == 1), z3.If(val_B, bit1 == 1, bit0 == 1))
                res = z3.If(op[i] == op_code, op_expr, res)
            solver.add(v[idx] == res)
            
        solver.add(v[-1] == bool(target_val))

    print("Solving...")
    if solver.check() == z3.sat:
        print(f"Solved in {time.time() - start:.2f} seconds!")
    else:
        print("Failed!")

test_scaling(1)
test_scaling(2)
