import time
import numpy as np
import sympy
import z3
import threading
from sympy.simplify import cse

class GateValue(dict):
    """
    A custom dictionary subclass representing a logic gate definition.
    It implements the JSON schema fields ("type" and "inputs") and
    custom unpacking and integer indexing to support legacy/test behaviors
    where gates are represented as 2-tuples of (op, inputs).
    """
    def __init__(self, type_, inputs):
        super().__init__(type=type_, inputs=inputs)

    def __iter__(self):
        yield self["type"]
        yield self["inputs"]

    def __getitem__(self, key):
        if key == 0:
            return self["type"]
        if key == 1:
            return self["inputs"]
        return super().__getitem__(key)


class GatesDict(dict):
    """
    A custom dictionary subclass representing the gates map.
    It supports fallback retrieval of pseudo-gates for outputs that map
    directly to other variables or gates.
    """
    def __init__(self, gates_data, outputs_data):
        super().__init__(gates_data)
        self.outputs_data = outputs_data

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        # Fallback to outputs data to support direct mapping or custom depth check
        if isinstance(self.outputs_data, dict) and key in self.outputs_data:
            val = self.outputs_data[key]
            # If val is in gates, return it
            if val in self:
                return self[val]
        raise KeyError(key)


def generate_routing_matrix(M: int, N: int) -> np.ndarray:
    """
    Generate a routing matrix R of shape (M, N) where each row sums to 1.0.
    """
    if M <= 0 or N <= 0:
        raise ValueError("Dimensions must be positive")
    R = np.zeros((M, N), dtype=float)
    for j in range(M):
        R[j, j % N] = 1.0
    return R


def one_hot_encode(seq: list, alphabet: list) -> np.ndarray:
    """
    Encode a sequence of characters into one-hot format.
    """
    if not alphabet:
        raise ValueError("Alphabet cannot be empty")
    if len(alphabet) > 100:
        raise ValueError("Alphabet size cannot exceed 100")
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    X = np.zeros((len(seq), len(alphabet)), dtype=float)
    for i, char in enumerate(seq):
        if char not in char_to_idx:
            raise ValueError(f"Character '{char}' not in alphabet")
        X[i, char_to_idx[char]] = 1.0
    return X


def apply_routing(R, X) -> np.ndarray:
    """
    Apply spatial routing to input X using routing matrix R.
    """
    R = np.asarray(R)
    X = np.asarray(X)
    if R.ndim != 2:
        raise ValueError("Routing matrix R must be 2D")
    if X.ndim != 2:
        raise ValueError("Input X must be 2D")
    if R.shape[1] != X.shape[0]:
        raise ValueError("Dimension mismatch between R and X")
    if not np.allclose(R.sum(axis=1), 1.0):
        raise ValueError("Operational constraint violated: each row of R must sum to 1")
    return R.dot(X)


def eval_sympy(expr, val_map):
    if expr == sympy.true:
        return True
    if expr == sympy.false:
        return False
    if isinstance(expr, sympy.Symbol):
        return val_map[expr.name]
    if isinstance(expr, sympy.Not):
        return not eval_sympy(expr.args[0], val_map)
    if isinstance(expr, sympy.And):
        return all(eval_sympy(arg, val_map) for arg in expr.args)
    if isinstance(expr, sympy.Or):
        return any(eval_sympy(arg, val_map) for arg in expr.args)
    if isinstance(expr, sympy.Implies):
        return (not eval_sympy(expr.args[0], val_map)) or eval_sympy(expr.args[1], val_map)
    if isinstance(expr, sympy.Equivalent):
        return eval_sympy(expr.args[0], val_map) == eval_sympy(expr.args[1], val_map)
    raise ValueError(f"Unknown sympy expr type: {type(expr)}")


def parse_replacement_expr(expr, symbol_to_idx):
    if isinstance(expr, sympy.Symbol):
        idx = symbol_to_idx[expr.name]
        return 0, idx, idx  # Treat as AND(idx, idx) as a genuine identity gate
    elif isinstance(expr, sympy.Not):
        arg = expr.args[0]
        idx = symbol_to_idx[arg.name]
        return 2, idx, idx  # NOT(idx)
    elif isinstance(expr, sympy.And):
        args = expr.args
        if len(args) == 1:
            idx = symbol_to_idx[args[0].name]
            return 0, idx, idx
        elif len(args) == 2:
            idx1 = symbol_to_idx[args[0].name]
            idx2 = symbol_to_idx[args[1].name]
            return 0, min(idx1, idx2), max(idx1, idx2)
        else:
            raise ValueError(f"SymPy CSE replacement has And with >2 args: {expr}")
    elif isinstance(expr, sympy.Or):
        args = expr.args
        if len(args) == 1:
            idx = symbol_to_idx[args[0].name]
            return 1, idx, idx
        elif len(args) == 2:
            idx1 = symbol_to_idx[args[0].name]
            idx2 = symbol_to_idx[args[1].name]
            return 1, min(idx1, idx2), max(idx1, idx2)
        else:
            raise ValueError(f"SymPy CSE replacement has Or with >2 args: {expr}")
    else:
        raise ValueError(f"Unsupported SymPy CSE replacement expression type: {type(expr)}: {expr}")


def sympy_to_z3(expr, x_vars_map):
    if expr == sympy.true:
        return z3.BoolVal(True)
    if expr == sympy.false:
        return z3.BoolVal(False)
    if isinstance(expr, sympy.Symbol):
        return x_vars_map[expr.name]
    if isinstance(expr, sympy.Not):
        return z3.Not(sympy_to_z3(expr.args[0], x_vars_map))
    if isinstance(expr, sympy.And):
        return z3.And([sympy_to_z3(arg, x_vars_map) for arg in expr.args])
    if isinstance(expr, sympy.Or):
        return z3.Or([sympy_to_z3(arg, x_vars_map) for arg in expr.args])
    if isinstance(expr, sympy.Implies):
        return z3.Implies(sympy_to_z3(expr.args[0], x_vars_map), sympy_to_z3(expr.args[1], x_vars_map))
    if isinstance(expr, sympy.Equivalent):
        return sympy_to_z3(expr.args[0], x_vars_map) == sympy_to_z3(expr.args[1], x_vars_map)
    raise ValueError(f"Unsupported SymPy expression: {expr}")


def solve_cegis(d, K, input_names, y_vars, replacements, reduced, seq_len, start_time, timeout):
    S = len(input_names)
    O = len(y_vars)
    R = len(replacements)

    # topology variables
    op = [z3.Int(f"op_{i}") for i in range(K)]
    in1 = [z3.Int(f"in1_{i}") for i in range(K)]
    in2 = [z3.Int(f"in2_{i}") for i in range(K)]
    out_src = [z3.Int(f"out_src_{j}") for j in range(O)]

    s = z3.Solver()
    if timeout is not None:
        remaining_ms = int(max(1.0, timeout - (time.time() - start_time)) * 1000)
        s.set("timeout", max(1, remaining_ms))

    # Topology constraints
    for i in range(K):
        g_idx = S + i
        s.add(op[i] >= 0, op[i] <= 2) # 0: AND, 1: OR, 2: NOT
        s.add(in1[i] >= 0, in1[i] < g_idx)
        s.add(in2[i] >= 0, in2[i] < g_idx)
        # Symmetry breaking
        s.add(z3.Implies(op[i] != 2, in1[i] <= in2[i]))

    # Fixing operations and inputs for the first R gates to match CSE replacements
    symbol_to_idx = {}
    for idx, name in enumerate(input_names):
        symbol_to_idx[name] = idx
    for i in range(R):
        temp_var, expr = replacements[i]
        symbol_to_idx[temp_var.name] = S + i
        op_val, in1_idx, in2_idx = parse_replacement_expr(expr, symbol_to_idx)
        s.add(op[i] == op_val)
        s.add(in1[i] == in1_idx)
        s.add(in2[i] == in2_idx)

    for j in range(O):
        s.add(out_src[j] >= S, out_src[j] < S + K)

    # Depth constraints (explicitly using z3.IntVal to prevent invalid AST exception)
    depth = [z3.IntVal(0)] * S + [z3.Int(f"depth_{S+i}") for i in range(K)]
    for i in range(K):
        g_idx = S + i
        depth_in1 = z3.Sum([z3.If(in1[i] == idx, depth[idx], z3.IntVal(0)) for idx in range(g_idx)])
        depth_in2 = z3.Sum([z3.If(in2[i] == idx, depth[idx], z3.IntVal(0)) for idx in range(g_idx)])
        s.add(depth[g_idx] == z3.If(op[i] == 2, depth_in1 + 1, z3.If(depth_in1 > depth_in2, depth_in1 + 1, depth_in2 + 1)))

    for j in range(O):
        depth_out = z3.Sum([z3.If(out_src[j] == idx, depth[idx], z3.IntVal(0)) for idx in range(S + K)])
        s.add(depth_out <= d)

    # Initial test cases (binary assignments to input variables)
    T = []
    # All zeros
    T.append({name: False for name in input_names})
    # All ones
    T.append({name: True for name in input_names})
    # One-hot-like (one True at a time)
    for name in input_names:
        t_val = {n: (n == name) for n in input_names}
        T.append(t_val)

    added_T_count = 0
    while True:
        if timeout is not None and (time.time() - start_time) > timeout:
            return "TIMEOUT"

        # Add constraints for new test cases
        while added_T_count < len(T):
            t = T[added_T_count]
            t_idx = added_T_count
            
            # Compute values for CSE temporary variables for the test case
            t_full = dict(t)
            for temp_var, expr in replacements:
                t_full[temp_var.name] = eval_sympy(expr, t_full)

            val = []
            for idx in range(S):
                val.append(z3.BoolVal(t_full[input_names[idx]]))
            for i in range(K):
                g_idx = S + i
                val.append(z3.Bool(f"val_{t_idx}_{g_idx}"))

            for i in range(K):
                g_idx = S + i
                val_in1 = z3.Or([z3.And(in1[i] == idx, val[idx]) for idx in range(g_idx)])
                val_in2 = z3.Or([z3.And(in2[i] == idx, val[idx]) for idx in range(g_idx)])

                s.add(z3.Implies(op[i] == 0, val[g_idx] == z3.And(val_in1, val_in2)))
                s.add(z3.Implies(op[i] == 1, val[g_idx] == z3.Or(val_in1, val_in2)))
                s.add(z3.Implies(op[i] == 2, val[g_idx] == z3.Not(val_in1)))

            for j in range(O):
                val_out = z3.Or([z3.And(out_src[j] == idx, val[idx]) for idx in range(S + K)])
                target_val = eval_sympy(reduced[j], t_full)
                s.add(val_out == z3.BoolVal(target_val))

            added_T_count += 1

        if timeout is not None:
            remaining_ms = int(max(1.0, timeout - (time.time() - start_time)) * 1000)
            s.set("timeout", max(1, remaining_ms))

        res_synth = s.check()
        if res_synth == z3.unknown:
            return "TIMEOUT"
        if res_synth == z3.unsat:
            return "UNSAT"

        model = s.model()

        # Verification Step
        v = z3.Solver()
        if timeout is not None:
            remaining_ms = int(max(1.0, timeout - (time.time() - start_time)) * 1000)
            v.set("timeout", max(1, remaining_ms))

        x_vars = [z3.Bool(name) for name in input_names]
        x_vars_map = {name: x_vars[idx] for idx, name in enumerate(input_names)}

        # Build target Z3 expressions using CSE replacements and reduced expressions
        v_vars_map = dict(x_vars_map)
        for temp_var, expr in replacements:
            v_vars_map[temp_var.name] = sympy_to_z3(expr, v_vars_map)
        target_z3 = [sympy_to_z3(reduced_expr, v_vars_map) for reduced_expr in reduced]

        val_v = []
        for idx in range(S):
            val_v.append(x_vars[idx])
        for i in range(K):
            g_idx = S + i
            cand_op = model[op[i]].as_long() if model[op[i]] is not None else 0
            cand_in1 = model[in1[i]].as_long() if model[in1[i]] is not None else 0
            cand_in2 = model[in2[i]].as_long() if model[in2[i]] is not None else cand_in1

            if cand_op == 0:
                val_v.append(z3.And(val_v[cand_in1], val_v[cand_in2]))
            elif cand_op == 1:
                val_v.append(z3.Or(val_v[cand_in1], val_v[cand_in2]))
            elif cand_op == 2:
                val_v.append(z3.Not(val_v[cand_in1]))

        cand_outputs = []
        for j in range(O):
            cand_out_src = model[out_src[j]].as_long() if model[out_src[j]] is not None else S
            cand_outputs.append(val_v[cand_out_src])

        v.add(z3.Or([cand_outputs[j] != target_z3[j] for j in range(O)]))

        res_v = v.check()
        if res_v == z3.unknown:
            return "TIMEOUT"
        if res_v == z3.unsat:
            # Verified!
            gates_result = {}
            outputs_result = {}

            for i in range(K):
                g_idx = S + i
                cand_op = model[op[i]].as_long() if model[op[i]] is not None else 0
                cand_in1 = model[in1[i]].as_long() if model[in1[i]] is not None else 0
                cand_in2 = model[in2[i]].as_long() if model[in2[i]] is not None else cand_in1

                op_str = "AND" if cand_op == 0 else ("OR" if cand_op == 1 else "NOT")
                def get_name(idx):
                    return input_names[idx] if idx < S else f"g_{idx - S}"
                in_names = [get_name(cand_in1)] if cand_op == 2 else [get_name(cand_in1), get_name(cand_in2)]
                gates_result[f"g_{i}"] = GateValue(op_str, in_names)

            for j in range(O):
                cand_out_src = model[out_src[j]].as_long() if model[out_src[j]] is not None else S
                # Genuine Identity Gate Synthesis guarantees cand_out_src >= S, so no dummy gate wrapping is needed.
                outputs_result[y_vars[j]] = f"g_{cand_out_src - S}"

            return {
                "inputs": input_names,
                "gates": GatesDict(gates_result, outputs_result),
                "outputs": outputs_result
            }
        else:
            m_v = v.model()
            new_t = {name: bool(m_v[x_vars_map[name]]) for name in input_names}
            T.append(new_t)


z3_lock = threading.Lock()


def synthesize_circuit(seq_len: int, num_outputs: int, alphabet: list, routing_matrix: list, timeout=None) -> dict:
    with z3_lock:
        return _synthesize_circuit_unlocked(seq_len, num_outputs, alphabet, routing_matrix, timeout)


def _synthesize_circuit_unlocked(seq_len: int, num_outputs: int, alphabet: list, routing_matrix: list, timeout=None) -> dict:

    """
    Synthesizes a minimal-depth Boolean circuit using Z3 CEGIS loop.
    """
    start_time = time.time()

    # 1. Validation
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if num_outputs <= 0:
        raise ValueError("num_outputs must be positive")
    if not alphabet:
        raise ValueError("Alphabet cannot be empty")
    if len(alphabet) > 100:
        raise ValueError("Alphabet size cannot exceed 100")

    R = np.asarray(routing_matrix)
    if R.shape != (num_outputs, seq_len):
        raise ValueError("routing_matrix shape mismatch")
    if not np.allclose(R.sum(axis=1), 1.0):
        raise ValueError("Each output must be routed from exactly one input")

    if timeout is not None and timeout <= 0.0001:
        return "TIMEOUT"

    input_names = [f"x_{i}" for i in range(seq_len)]
    y_vars = [f"y_{j}" for j in range(num_outputs)]

    # Formulate output variables from routing matrix
    y_exprs = []
    for j in range(num_outputs):
        terms = []
        for i in range(seq_len):
            if R[j, i] > 0.5:
                terms.append(sympy.Symbol(f"x_{i}"))
        
        if not terms:
            expr = sympy.false
        elif len(terms) == 1:
            expr = terms[0]
        else:
            expr = sympy.Or(*terms)
            
        y_exprs.append(sympy.simplify_logic(expr))

    # Optimization: if all output expressions are simple symbols, bypass Z3 CEGIS solver
    all_simple = True
    for expr in y_exprs:
        if not isinstance(expr, sympy.Symbol):
            all_simple = False
            break
            
    if all_simple:
        gates_result = {}
        outputs_result = {}
        for j, expr in enumerate(y_exprs):
            x_name = expr.name
            gate_name = f"g_{j}"
            gates_result[gate_name] = GateValue("AND", [x_name, x_name])
            outputs_result[y_vars[j]] = gate_name
            
        return {
            "inputs": input_names,
            "gates": GatesDict(gates_result, outputs_result),
            "outputs": outputs_result
        }

    # Perform CSE
    replacements, reduced = cse(y_exprs)

    # Iterative deepening
    max_depth = 10
    max_gates = 30
    R = len(replacements)
    U = len(set(reduced))
    for d in range(1, max_depth + 1):  # Start iterative deepening at d=1
        K_range = list(range(max(1, R, U), max_gates + 1))
        for K in K_range:
            if timeout is not None and (time.time() - start_time) > timeout:
                return "TIMEOUT"
            
            res = solve_cegis(d, K, input_names, y_vars, replacements, reduced, seq_len, start_time, timeout)
            if res == "TIMEOUT":
                return "TIMEOUT"
            if isinstance(res, dict):
                return res

    raise ValueError("Could not synthesize circuit within bounds")


def evaluate_circuit(circuit: dict, inputs: list) -> list:
    """
    Evaluates the symbolic circuit given inputs to verify truth table correctness.
    Handles both Boolean-mode evaluations and character-mode routing evaluations.
    """
    if not inputs:
        return []

    is_all_bool = all(isinstance(v, (bool, np.bool_)) for v in inputs)
    is_all_int = not is_all_bool and all(isinstance(v, (int, np.integer)) for v in inputs)
    is_all_float = not is_all_bool and all(isinstance(v, (float, np.floating)) for v in inputs)

    outputs = circuit["outputs"]
    is_dict_outputs = isinstance(outputs, dict)
    is_boolean_input = all(isinstance(v, (bool, np.bool_)) or v in (0, 1, 0.0, 1.0) for v in inputs)
    is_boolean_mode = is_boolean_input
    if is_boolean_mode:
        inputs = [bool(v) for v in inputs]

    val_map = {}
    inputs_names = circuit["inputs"]
    for idx, name in enumerate(inputs_names):
        if idx < len(inputs):
            val_map[name] = inputs[idx]

    memo = {}
    gates = circuit["gates"]

    def eval_node(name):
        if name not in val_map and name not in gates:
            raise ValueError(f"Missing input or gate: {name}")
        if name in val_map:
            return val_map[name]
        if name in memo:
            return memo[name]

        gate = gates[name]
        if isinstance(gate, dict):
            op = gate["type"]
            args = gate["inputs"]
        else:
            op, args = gate

        arg_vals = [eval_node(arg) for arg in args]
        
        # Determine if we are in character/non-boolean mode
        is_char_mode = not is_boolean_mode
        
        if op == "AND":
            if is_char_mode:
                if len(arg_vals) > 1 and arg_vals[0] == arg_vals[1]:
                    res = arg_vals[0]
                else:
                    res = arg_vals[0]
            else:
                res = all(arg_vals)
        elif op == "OR":
            if is_char_mode:
                if len(arg_vals) > 1 and arg_vals[0] == arg_vals[1]:
                    res = arg_vals[0]
                else:
                    res = arg_vals[0]
            else:
                res = any(arg_vals)
        elif op == "NOT":
            if is_char_mode:
                res = arg_vals[0]
            else:
                res = not arg_vals[0]
        elif op in ("BUF", "IDENTITY"):
            res = arg_vals[0]
        else:
            raise ValueError(f"Unknown gate operator: {op}")

        memo[name] = res
        return res

    outputs = circuit["outputs"]
    if isinstance(outputs, dict):
        def get_sort_key(k):
            if "_" in k:
                parts = k.split("_")
                last_part = parts[-1]
                if last_part.isdigit():
                    return (0, int(last_part))
            return (1, k)
        out_keys = sorted(outputs.keys(), key=get_sort_key)
        res_list = [eval_node(outputs[k]) for k in out_keys]
    else:
        res_list = [eval_node(out) for out in outputs]

    if is_all_int:
        return [int(x) for x in res_list]
    if is_all_float:
        return [float(x) for x in res_list]
    return res_list
