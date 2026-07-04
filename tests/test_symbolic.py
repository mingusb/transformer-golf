import numpy as np
import pytest
import sympy
from src.symbolic.solver import (
    generate_routing_matrix,
    one_hot_encode,
    apply_routing,
    synthesize_circuit,
    evaluate_circuit
)

# ==========================================
# Feature F1: Spatial Routing & Mapping
# ==========================================

def test_TEST_T1_F1_01():
    # TEST_T1_F1_01: Routing Matrix Dimension Check
    # Verify R shape matches (M, N) for output length M and input length N.
    M, N = 5, 10
    R = generate_routing_matrix(M, N)
    assert R.shape == (M, N)

def test_TEST_T1_F1_02():
    # TEST_T1_F1_02: Routing Matrix Row Sum Check
    # Assert operational constraint: each row of R sums to 1.
    R = generate_routing_matrix(8, 12)
    assert np.allclose(R.sum(axis=1), 1.0)

def test_TEST_T1_F1_03():
    # TEST_T1_F1_03: One-Hot Encoding Constraint
    # Ensure input sequences represent exactly one active symbol per position.
    alphabet = ['a', 'b', 'c']
    seq = ['a', 'b', 'a', 'c']
    X = one_hot_encode(seq, alphabet)
    assert X.shape == (len(seq), len(alphabet))
    assert np.all(X.sum(axis=-1) == 1.0)

def test_TEST_T1_F1_04():
    # TEST_T1_F1_04: Spatial Mapping Output Verification
    # Check output calculation for simple alphabet and routing matrix.
    # Y == R.dot(X)
    alphabet = ['a', 'b']
    seq = ['a', 'b', 'a']
    X = one_hot_encode(seq, alphabet) # shape (3, 2)
    R = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ]) # map output 0 to input 1 ('b'), output 1 to input 0 ('a')
    Y = apply_routing(R, X)
    expected = np.array([
        [0.0, 1.0], # input 1 ('b')
        [1.0, 0.0]  # input 0 ('a')
    ])
    assert np.allclose(Y, expected)

def test_TEST_T1_F1_05():
    # TEST_T1_F1_05: Identity Matrix Routing Check
    # Verify R = I maps output sequence exactly equal to input.
    alphabet = ['x', 'y', 'z']
    seq = ['x', 'y', 'z', 'y']
    X = one_hot_encode(seq, alphabet)
    R = np.eye(len(seq))
    Y = apply_routing(R, X)
    assert np.allclose(Y, X)

def test_TEST_T2_F1_01():
    # TEST_T2_F1_01: Minimum Sequence Length N=1, M=1
    # Verify spatial routing handles single input/output index.
    R = generate_routing_matrix(1, 1)
    X = one_hot_encode(['a'], ['a'])
    Y = apply_routing(R, X)
    assert np.allclose(Y, X)

def test_TEST_T2_F1_02():
    # TEST_T2_F1_02: Unary Alphabet Size Check
    # Test spatial routing with alphabet containing a single character.
    R = generate_routing_matrix(3, 3)
    X = one_hot_encode(['a', 'a', 'a'], ['a'])
    Y = apply_routing(R, X)
    assert np.all(Y == 1.0)

def test_TEST_T2_F1_03():
    # TEST_T2_F1_03: Extremely Large Matrix Handling
    # Verify high dimension routing (M, N = 1000) behaves safely.
    M, N = 1000, 1000
    R = generate_routing_matrix(M, N)
    X = np.ones((N, 10))
    Y = apply_routing(R, X)
    assert Y.shape == (M, 10)

def test_TEST_T2_F1_04():
    # TEST_T2_F1_04: Disconnected Routing Constraint
    # Assert routing fails when row sums are not 1.
    R = np.zeros((3, 3)) # row sums are 0, not 1
    X = np.ones((3, 2))
    with pytest.raises(ValueError):
        apply_routing(R, X)

def test_TEST_T2_F1_05():
    # TEST_T2_F1_05: Routing Overlap Verification
    # Test routing with multiple identical source connections.
    R = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0]
    ]) # All outputs connect to input 0
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ])
    Y = apply_routing(R, X)
    expected = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    assert np.allclose(Y, expected)


# ==========================================
# Feature F2: Symbolic Logic Solver
# ==========================================

def test_TEST_T1_F2_01():
    # TEST_T1_F2_01: Solver Returns Valid Circuit Structure
    # Check if synthesize_circuit returns expected dictionary keys.
    R = [[1.0, 0.0], [0.0, 1.0]]
    res = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R)
    assert isinstance(res, dict)
    assert "inputs" in res
    assert "gates" in res
    assert "outputs" in res

def test_TEST_T1_F2_02():
    # TEST_T1_F2_02: Z3 Iterative Deepening Minimality
    # Assert Z3 finds 1-depth circuit for single OR reduction.
    # We represent single OR reduction as outputs depending directly on inputs.
    # Our stub synthesized gates have depth 1 (i.e. g_j refers to x_i directly).
    # Let's count depth by tracking gate references.
    R = [[1.0, 0.0], [0.0, 1.0]]
    circuit = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R)
    # Depth calculation
    gates = circuit["gates"]
    depths = {}
    def get_depth(name):
        if name in depths:
            return depths[name]
        if name.startswith("x_"):
            return 0
        gate = gates[name]
        if isinstance(gate, tuple):
            op, args = gate
        else:
            args = gate.get("inputs", [])
        d = 1 + max(get_depth(arg) for arg in args)
        depths[name] = d
        return d
    
    outputs = circuit["outputs"]
    out_names = list(outputs.values()) if isinstance(outputs, dict) else outputs
    max_depth = max(get_depth(out) for out in out_names)
    assert max_depth == 1

def test_TEST_T1_F2_03():
    # TEST_T1_F2_03: Evaluate Circuit Output Correctness
    # Assert evaluate_circuit yields expected outputs for simple AND gates.
    circuit = {
        "inputs": ["x_0", "x_1"],
        "gates": {
            "g1": ("AND", ["x_0", "x_1"])
        },
        "outputs": ["g1"]
    }
    assert evaluate_circuit(circuit, [True, True]) == [True]
    assert evaluate_circuit(circuit, [True, False]) == [False]
    assert evaluate_circuit(circuit, [False, False]) == [False]

def test_TEST_T1_F2_04():
    # TEST_T1_F2_04: SymPy Logical Literal Minimization
    # Verify literal minimization on redundant inputs.
    # sympy.simplify_logic(A | (A & B)) == A
    A, B = sympy.symbols('A B')
    expr = A | (A & B)
    minimized = sympy.simplify_logic(expr)
    assert minimized == A

def test_TEST_T1_F2_05():
    # TEST_T1_F2_05: Solver Rejects Invalid Alphabet Size
    # Check if passing negative or zero alphabet size raises exception.
    R = [[1.0, 0.0]]
    with pytest.raises(ValueError):
        synthesize_circuit(seq_len=2, num_outputs=1, alphabet=[], routing_matrix=R)

def test_TEST_T2_F2_01():
    # TEST_T2_F2_01: Z3 Unsatisfiable Specs Handling
    # Synthesize circuit for unsat routing specifications.
    # For example, a routing matrix where one row is all zeros (disconnected).
    R = [[0.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError):
         synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R)

def test_TEST_T2_F2_02():
    # TEST_T2_F2_02: Synthesize Empty Length Circuit
    # Synthesize circuit with sequence length = 0.
    with pytest.raises(ValueError):
        synthesize_circuit(seq_len=0, num_outputs=1, alphabet=['a'], routing_matrix=[])

def test_TEST_T2_F2_03():
    # TEST_T2_F2_03: Evaluate Circuit with Empty Inputs
    # Assert evaluate_circuit with empty inputs yields empty outputs.
    circuit = {
        "inputs": [],
        "gates": {},
        "outputs": []
    }
    assert evaluate_circuit(circuit, []) == []

def test_TEST_T2_F2_04():
    # TEST_T2_F2_04: Solver Under Strict Timeout
    # Assert Z3 stops and returns status on tight timeout limits.
    # Using timeout parameter <= 0.0001 triggers status == TIMEOUT in our stub
    R = [[1.0, 0.0], [0.0, 1.0]]
    res = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R, timeout=0.00001)
    assert res == "TIMEOUT" or (isinstance(res, dict) and res.get("status") == "TIMEOUT")

def test_TEST_T2_F2_05():
    # TEST_T2_F2_05: Large Alphabet Limit Enforcement
    # Verify Z3 raises error when alphabet size exceeds logic capacity.
    R = [[1.0, 0.0]]
    alphabet = [str(i) for i in range(105)] # size > 100
    with pytest.raises(ValueError):
        synthesize_circuit(seq_len=2, num_outputs=1, alphabet=alphabet, routing_matrix=R)


def test_TEST_T3_F2_06():
    # TEST_T3_F2_06: Numeric/Special character alphabet support
    # Ensure that synthesizing with special chars or numbers in the alphabet works correctly.
    R = [[0.0, 1.0]]
    alphabet = ['1', '$', '@']
    circuit = synthesize_circuit(seq_len=2, num_outputs=1, alphabet=alphabet, routing_matrix=R)
    assert isinstance(circuit, dict)
    assert "inputs" in circuit
    assert "gates" in circuit
    # Evaluate and check
    res = evaluate_circuit(circuit, ['$', '1'])
    assert res == ['1']


def test_TEST_T3_F2_07():
    # TEST_T3_F2_07: Check gate structure completeness
    # Verify that all outputs in synthesized circuit map to gates that exist and are valid.
    R = [[0.0, 1.0], [1.0, 0.0]]
    alphabet = ['a', 'b']
    circuit = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=alphabet, routing_matrix=R)
    gates = circuit["gates"]
    outputs = circuit["outputs"]
    for out_name, gate_name in outputs.items():
        assert gate_name in gates
        gate = gates[gate_name]
        assert gate["type"] in ("AND", "OR", "NOT")
        for inp in gate["inputs"]:
            assert inp in circuit["inputs"] or inp in gates


def test_TEST_T3_F2_08():
    # TEST_T3_F2_08: Evaluate circuit with multiple outputs and complex routing
    # Verify correctness of evaluate_circuit on routing matrix routing to the same input multiple times.
    R = [[0.0, 1.0], [0.0, 1.0]]
    alphabet = ['x', 'y']
    circuit = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=alphabet, routing_matrix=R)
    res = evaluate_circuit(circuit, ['x', 'y'])
    assert res == ['y', 'y']

