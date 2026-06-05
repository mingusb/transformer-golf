import z3
import json

def synthesize_bool_eq():
    print("Synthesizing optimal bool_eq (XNOR) gate...")
    # Inputs: a, b
    # Target: (a & b) | ((~a & 1) & (~b & 1))
    
    for N in range(1, 4):
        solver = z3.Solver()
        op = [z3.Int(f'op_{i}') for i in range(2, 2+N)]
        in_A = [z3.Int(f'inA_{i}') for i in range(2, 2+N)]
        in_B = [z3.Int(f'inB_{i}') for i in range(2, 2+N)]
        
        for i in range(N):
            idx = 2 + i
            solver.add(op[i] >= 0, op[i] <= 15)
            solver.add(in_A[i] >= 0, in_A[i] < idx)
            solver.add(in_B[i] >= 0, in_B[i] < idx)
            
        for state in range(4):
            a_val = bool((state >> 0) & 1)
            b_val = bool((state >> 1) & 1)
            
            import sys
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from scripts.task_spec import bitwise_match_logic
            target_val = bitwise_match_logic(a_val, b_val)
            
            v = [z3.Bool(f'v_{idx}_{state}') for idx in range(2+N)]
            solver.add(v[0] == a_val)
            solver.add(v[1] == b_val)
            
            for i in range(N):
                idx = 2 + i
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
            solver.add(v[-1] == target_val)
            
        if solver.check() == z3.sat:
            m = solver.model()
            gates = []
            for i in range(N):
                idx = 2 + i
                op_val = m.evaluate(op[i]).as_long()
                in_A_val = m.evaluate(in_A[i]).as_long()
                in_B_val = m.evaluate(in_B[i]).as_long()
                
                # Convert in_A_val to input names
                def get_name(v):
                    if v == 0: return "a"
                    if v == 1: return "b"
                    return f"gate_{v}"
                    
                gates.append({
                    "id": f"gate_{idx}",
                    "op": op_val,
                    "in_a": get_name(in_A_val),
                    "in_b": get_name(in_B_val)
                })
            return gates

if __name__ == '__main__':
    gates = synthesize_bool_eq()
    
    op_py_map = {
        0: "0", 1: "~({a} | {b}) & 1", 2: "{a} & ~{b} & 1", 3: "~{b} & 1",
        4: "~{a} & {b} & 1", 5: "~{a} & 1", 6: "{a} ^ {b}", 7: "~({a} & {b}) & 1",
        8: "{a} & {b}", 9: "~({a} ^ {b}) & 1", 10: "{b}", 11: "{a} | ~{b} & 1",
        12: "{a}", 13: "~{a} | {b} & 1", 14: "{a} | {b}", 15: "1"
    }
    
    # Generate python code
    lines = []
    lines.append("def bool_eq(a, b):")
    lines.append("    # Z3 Mathematically Superoptimized Gate")
    for g in gates:
        expr = op_py_map[g['op']].format(a=g['in_a'], b=g['in_b'])
        if g == gates[-1]:
            lines.append(f"    return {expr}")
        else:
            lines.append(f"    {g['id']} = {expr}")
            
    py_code = "\n".join(lines)
    print("Synthesized Python Code:")
    print(py_code)
    
    depth = max(1, len(gates))
    with open("../docs/z3_ast.json", "w") as f:
        json.dump({
            "python_code": py_code, 
            "mlp_layers": depth,
            "raw_ast": gates
        }, f, indent=2)
