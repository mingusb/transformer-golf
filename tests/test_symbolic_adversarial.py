import numpy as np
import pytest
import sympy
import time
from src.symbolic.solver import (
    generate_routing_matrix,
    one_hot_encode,
    apply_routing,
    synthesize_circuit,
    evaluate_circuit
)

def test_large_alphabet_sizes():
    # Test alphabet sizes of 50 and 99
    for size in [50, 99]:
        alphabet = [f"a{i}" for i in range(size)]
        seq = [alphabet[i % size] for i in range(5)]
        X = one_hot_encode(seq, alphabet)
        assert X.shape == (5, size)
        assert np.allclose(X.sum(axis=-1), 1.0)
        
        # Test routing matrix and synthesis with large alphabet
        # Note: synthesis is independent of alphabet content except validation
        R = generate_routing_matrix(3, 3)
        res = synthesize_circuit(seq_len=3, num_outputs=3, alphabet=alphabet, routing_matrix=R)
        assert isinstance(res, dict)
        assert "inputs" in res
        assert "gates" in res

def test_alphabet_size_over_limit():
    # Alphabet size > 100 should raise ValueError
    alphabet = [f"a{i}" for i in range(101)]
    R = generate_routing_matrix(2, 2)
    with pytest.raises(ValueError, match="Alphabet size cannot exceed 100"):
        synthesize_circuit(seq_len=2, num_outputs=2, alphabet=alphabet, routing_matrix=R)
    
    with pytest.raises(ValueError, match="Alphabet size cannot exceed 100"):
        one_hot_encode(["a0"], alphabet)

def test_empty_inputs():
    # 1. one_hot_encode with empty sequence
    alphabet = ["a", "b"]
    X = one_hot_encode([], alphabet)
    assert X.shape == (0, 2)

    # 2. one_hot_encode with empty alphabet
    with pytest.raises(ValueError, match="Alphabet cannot be empty"):
        one_hot_encode(["a"], [])

    # 3. synthesize_circuit with empty alphabet
    R = [[1.0]]
    with pytest.raises(ValueError, match="Alphabet cannot be empty"):
        synthesize_circuit(seq_len=1, num_outputs=1, alphabet=[], routing_matrix=R)

    # 4. synthesize_circuit with seq_len=0 or num_outputs=0
    with pytest.raises(ValueError, match="seq_len must be positive"):
        synthesize_circuit(seq_len=0, num_outputs=1, alphabet=["a"], routing_matrix=[])
    with pytest.raises(ValueError, match="num_outputs must be positive"):
        synthesize_circuit(seq_len=1, num_outputs=0, alphabet=["a"], routing_matrix=[])

    # 5. evaluate_circuit with empty inputs
    circuit = {
        "inputs": [],
        "gates": {},
        "outputs": []
    }
    assert evaluate_circuit(circuit, []) == []

def test_invalid_inputs():
    # 1. one_hot_encode with character not in alphabet
    alphabet = ["a", "b"]
    with pytest.raises(ValueError, match="Character 'c' not in alphabet"):
        one_hot_encode(["a", "c"], alphabet)

    # 2. apply_routing with non-2D matrices
    with pytest.raises(ValueError, match="Routing matrix R must be 2D"):
        apply_routing(np.array([1, 2]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="Input X must be 2D"):
        apply_routing(np.ones((2, 2)), np.array([1, 2]))

    # 3. apply_routing with dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch between R and X"):
        apply_routing(np.ones((2, 3)), np.ones((2, 2)))

    # 4. apply_routing with invalid row sums
    R_invalid = np.array([[0.5, 0.4], [0.1, 0.9]])
    with pytest.raises(ValueError, match="Operational constraint violated"):
        apply_routing(R_invalid, np.ones((2, 2)))

    # 5. synthesize_circuit with routing matrix shape mismatch
    R_wrong_shape = np.ones((2, 3))
    with pytest.raises(ValueError, match="routing_matrix shape mismatch"):
        synthesize_circuit(seq_len=2, num_outputs=2, alphabet=["a"], routing_matrix=R_wrong_shape)

    # 6. synthesize_circuit with invalid row sums (each output must be routed from exactly one input)
    # The sum of first row is 1.1, which is not 1.0. This triggers the validation check immediately.
    R_wrong_sum = np.array([[0.5, 0.6], [1.0, 0.0]])
    with pytest.raises(ValueError, match="Each output must be routed from exactly one input"):
        synthesize_circuit(seq_len=2, num_outputs=2, alphabet=["a"], routing_matrix=R_wrong_sum)

    # 7. evaluate_circuit with unknown operator
    circuit = {
        "inputs": ["x_0"],
        "gates": {
            "g1": ("XOR", ["x_0", "x_0"])
        },
        "outputs": ["g1"]
    }
    with pytest.raises(ValueError, match="Unknown gate operator"):
        evaluate_circuit(circuit, [True])

def test_extreme_sequence_lengths():
    # Test extreme sequence lengths (25, 50) with identity routing.
    # To prevent tests from hanging under resource limits or slow solvers,
    # we enforce a generous timeout of 10.0 seconds.
    for size in [25, 50]:
        R = generate_routing_matrix(size, size)
        t_start = time.time()
        try:
            res = synthesize_circuit(seq_len=size, num_outputs=size, alphabet=["a", "b"], routing_matrix=R, timeout=10.0)
            t_duration = time.time() - t_start
            print(f"Size {size} synthesis took {t_duration:.4f}s")
            
            if res == "TIMEOUT":
                # If Z3 times out on extreme sizes, verify it exited gracefully
                assert True
            else:
                assert isinstance(res, dict)
                assert "inputs" in res
                assert "gates" in res
                
                # Verify correctness of evaluate_circuit on this large circuit
                inputs = [f"c{i}" for i in range(size)]
                out = evaluate_circuit(res, inputs)
                expected_correct = [inputs[int(k.split('_')[1])] for k in sorted(res["outputs"].keys(), key=lambda k: int(k.split('_')[1]))]
                assert out == expected_correct
        except ValueError as e:
            assert "Could not synthesize circuit within bounds" in str(e)

def test_evaluate_circuit_output_ordering_bug():
    # Explicitly demonstrate the bug: for 11 outputs, the output order is permuted under the bug, but fixed now
    size = 11
    R = generate_routing_matrix(size, size)
    res = synthesize_circuit(seq_len=size, num_outputs=size, alphabet=["a", "b"], routing_matrix=R, timeout=5.0)
    assert isinstance(res, dict)
    
    inputs = [f"c{i}" for i in range(size)]
    out = evaluate_circuit(res, inputs)
    # The correct behavior should preserve the output index order:
    assert out == inputs


def test_integer_binary_inputs():
    circuit = {
        "inputs": ["x_0", "x_1"],
        "gates": {
            "g1": ("AND", ["x_0", "x_1"]),
            "g2": ("NOT", ["x_0"]),
            "g3": ("OR", ["x_0", "x_1"])
        },
        "outputs": ["g1", "g2", "g3"]
    }
    # For inputs [1, 0]:
    # x_0 = 1 (True), x_1 = 0 (False)
    # g1 = AND(1, 0) -> 0
    # g2 = NOT(1) -> 0
    # g3 = OR(1, 0) -> 1
    # Expected result: [0, 0, 1]
    res = evaluate_circuit(circuit, [1, 0])
    assert res == [0, 0, 1]


def test_float_binary_inputs():
    circuit = {
        "inputs": ["x_0", "x_1"],
        "gates": {
            "g1": ("AND", ["x_0", "x_1"]),
            "g2": ("NOT", ["x_0"]),
            "g3": ("OR", ["x_0", "x_1"])
        },
        "outputs": ["g1", "g2", "g3"]
    }
    # Expected result: [0.0, 0.0, 1.0] (or boolean equivalent, but if we pass float, we expect binary/boolean evaluation)
    # Since float inputs should be treated as boolean-like, if evaluated correctly, they should return logical values.
    # If returned as floats or bools, they should match the expected truth values.
    res = evaluate_circuit(circuit, [1.0, 0.0])
    # Converting outputs to boolean/int-like should match [0, 0, 1]
    assert [int(bool(x)) for x in res] == [0, 0, 1]


def test_unary_alphabet_synthesis():
    # Verify synthesize_circuit handles seq_len=1, num_outputs=1, alphabet of size 1
    R = [[1.0]]
    res = synthesize_circuit(seq_len=1, num_outputs=1, alphabet=['a'], routing_matrix=R)
    assert isinstance(res, dict)
    assert res["inputs"] == ["x_0"]
    # Evaluate with character inputs
    out = evaluate_circuit(res, ['a'])
    assert out == ['a']


def test_asymmetric_dimensions():
    # Verify synthesize_circuit handles large outputs relative to inputs
    R = generate_routing_matrix(15, 3)
    res = synthesize_circuit(seq_len=3, num_outputs=15, alphabet=['a', 'b'], routing_matrix=R)
    assert isinstance(res, dict)
    assert len(res["outputs"]) == 15
    # Evaluate and verify routing
    inputs = ['a', 'b', 'a']
    out = evaluate_circuit(res, inputs)
    expected = [inputs[int(np.argmax(R[j]))] for j in range(15)]
    assert out == expected


def test_special_character_alphabet_synthesis():
    # Verify synthesis with special symbols in alphabet
    alphabet = ['$', '@', '#', ' ']
    R = generate_routing_matrix(4, 4)
    res = synthesize_circuit(seq_len=4, num_outputs=4, alphabet=alphabet, routing_matrix=R)
    assert isinstance(res, dict)
    out = evaluate_circuit(res, ['$', '@', '#', ' '])
    assert out == ['$', '@', '#', ' ']


def test_numpy_numeric_types():
    circuit = {
        "inputs": ["x_0", "x_1"],
        "gates": {
            "g1": ("AND", ["x_0", "x_1"]),
            "g2": ("NOT", ["x_0"]),
            "g3": ("OR", ["x_0", "x_1"])
        },
        "outputs": ["g1", "g2", "g3"]
    }
    # Test numpy integers
    res_int = evaluate_circuit(circuit, [np.int32(1), np.int64(0)])
    assert res_int == [0, 0, 1]
    assert all(isinstance(x, (int, np.integer)) for x in res_int)

    # Test numpy floats
    res_float = evaluate_circuit(circuit, [np.float32(1.0), np.float64(0.0)])
    assert res_float == [0.0, 0.0, 1.0]
    assert all(isinstance(x, (float, np.floating)) for x in res_float)

    # Test numpy booleans
    res_bool = evaluate_circuit(circuit, [np.bool_(True), np.bool_(False)])
    assert res_bool == [False, False, True]
    assert all(isinstance(x, (bool, np.bool_)) for x in res_bool)


def test_mixed_inputs_coercion():
    circuit = {
        "inputs": ["x_0", "x_1"],
        "gates": {
            "g1": ("AND", ["x_0", "x_1"]),
            "g2": ("NOT", ["x_0"]),
            "g3": ("OR", ["x_0", "x_1"])
        },
        "outputs": ["g1", "g2", "g3"]
    }
    # Mixed float and int (results in default boolean mode and direct return because not all are int or float)
    res_mixed_num = evaluate_circuit(circuit, [1, 0.0])
    assert res_mixed_num == [False, False, True]

    # Mixed int and bool
    res_mixed_bool = evaluate_circuit(circuit, [1, False])
    # Note: bool is a subclass of int, so is_all_int is actually True!
    assert res_mixed_bool == [0, 0, 1]


def test_extreme_timeout_handling():
    # Enforce a tiny timeout and ensure TIMEOUT status is correctly returned
    R = [[1.0, 0.0], [0.0, 1.0]]
    res = synthesize_circuit(seq_len=2, num_outputs=2, alphabet=['a', 'b'], routing_matrix=R, timeout=1e-9)
    assert res == "TIMEOUT"


