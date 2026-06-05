import z3

def superoptimize():
    print("Initializing Z3 SMT Solver for Boolean Superoptimization...")
    # Define inputs
    C1_0, C1_1, Q_0, Q_1, E = z3.Bools('C1_0 C1_1 Q_0 Q_1 E')
    
    # Define the target function (the raw unoptimized logic from the PyTorch model)
    raw_match_0 = z3.Or(z3.And(C1_0, Q_0), z3.And(z3.Not(C1_0), z3.Not(Q_0)))
    raw_match_1 = z3.Or(z3.And(C1_1, Q_1), z3.And(z3.Not(C1_1), z3.Not(Q_1)))
    target_func = z3.And(raw_match_0, z3.And(raw_match_1, E))
    
    print("Target Function (Raw PyTorch Logic):")
    print(target_func)
    print("\nStarting search for the absolute minimum gate-count circuit...")

    # We want to find a circuit using a minimal number of 2-input boolean gates.
    for N in range(1, 6): # Try circuit sizes 1 to 5
        print(f"Testing circuit size N = {N} gates...")
        solver = z3.Solver()
        
        # Variables for the circuit structure
        op = [z3.Int(f'op_{i}') for i in range(5, 5+N)]
        in_A = [z3.Int(f'inA_{i}') for i in range(5, 5+N)]
        in_B = [z3.Int(f'inB_{i}') for i in range(5, 5+N)]
        
        # Constraints on structure
        for i in range(N):
            idx = 5 + i
            solver.add(op[i] >= 0, op[i] <= 15)
            solver.add(in_A[i] >= 0, in_A[i] < idx)
            solver.add(in_B[i] >= 0, in_B[i] < idx)
            
        # Evaluate for all 32 truth table combinations
        for state in range(32):
            c1_0_val = bool((state >> 0) & 1)
            c1_1_val = bool((state >> 1) & 1)
            q_0_val = bool((state >> 2) & 1)
            q_1_val = bool((state >> 3) & 1)
            e_val = bool((state >> 4) & 1)
            
            def eval_func(c1_0, c1_1, q_0, q_1, e):
                m0 = (c1_0 and q_0) or (not c1_0 and not q_0)
                m1 = (c1_1 and q_1) or (not c1_1 and not q_1)
                return m0 and m1 and e
            target_val = eval_func(c1_0_val, c1_1_val, q_0_val, q_1_val, e_val)
            
            v = [z3.Bool(f'v_{idx}_{state}') for idx in range(5+N)]
            solver.add(v[0] == c1_0_val)
            solver.add(v[1] == c1_1_val)
            solver.add(v[2] == q_0_val)
            solver.add(v[3] == q_1_val)
            solver.add(v[4] == e_val)
            
            for i in range(N):
                idx = 5 + i
                
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
                    
                    op_expr = z3.If(val_A, 
                                    z3.If(val_B, bit3 == 1, bit2 == 1),
                                    z3.If(val_B, bit1 == 1, bit0 == 1))
                    res = z3.If(op[i] == op_code, op_expr, res)
                    
                solver.add(v[idx] == res)
            
            solver.add(v[5+N-1] == target_val)

        if solver.check() == z3.sat:
            print(f"\nSUCCESS! Found equivalent circuit with exactly {N} gates!")
            m = solver.model()
            op_names = {
                0: "FALSE", 1: "NOR", 2: "A AND NOT B", 3: "NOT B",
                4: "NOT A AND B", 5: "NOT A", 6: "XOR", 7: "NAND",
                8: "AND", 9: "XNOR", 10: "B", 11: "A OR NOT B",
                12: "A", 13: "NOT A OR B", 14: "OR", 15: "TRUE"
            }
            node_names = ["C1_0", "C1_1", "Q_0", "Q_1", "E"]
            for i in range(N):
                idx = 5 + i
                op_val = m.evaluate(op[i]).as_long()
                in_A_val = m.evaluate(in_A[i]).as_long()
                in_B_val = m.evaluate(in_B[i]).as_long()
                node_names.append(f"gate_{idx}")
                print(f"gate_{idx} = {op_names[op_val]}({node_names[in_A_val]}, {node_names[in_B_val]})")
            print("\nThis mathematical proof confirms the absolute minimum hardware gate-count!")
            return
            
    print("\nCould not find a circuit of size 5 or less.")

if __name__ == '__main__':
    superoptimize()
