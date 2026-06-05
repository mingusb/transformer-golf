import time
import pytest
import numpy as np
import sympy
import torch
from src.symbolic.solver import (
    synthesize_circuit,
    evaluate_circuit,
    solve_cegis
)
from src.models.map_reduce_mlp import compile_mlp_from_circuit
from src.data.dfa import DFAGenerator

def test_dfa_generator_type_mismatch():
    """
    Test DFAGenerator type mismatch:
    When alphabet contains integers, the transitions are checked with integers during __init__
    but compared against stringified values during generate_sequences.
    This causes transitions to silently fail and fall back to start_state.
    """
    # Define transition function with integer keys/values
    tf = {0: {0: 1, 1: 0}, 1: {0: 0, 1: 1}}
    generator = DFAGenerator(
        num_states=2,
        alphabet=[0, 1],
        transition_function=tf,
        start_state=0,
        accept_states=[1]
    )
    inputs, labels, paths = generator.generate_sequences(num_samples=10, seq_len=5)
    
    # Now that the bug is fixed, transitions work correctly, so paths should not
    # all be stuck at start_state (0) [0, 0, 0, 0, 0, 0].
    # There should be at least some transition to state 1.
    has_transitioned = False
    for path in paths:
        if 1 in path:
            has_transitioned = True
            break
    assert has_transitioned, "DFAGenerator should transition to state 1 since the transitions now work correctly."

def test_map_reduce_mlp_duplicate_inputs():
    """
    Test MapReduceMLP with duplicate inputs:
    If a circuit has duplicate input names, compile_mlp_from_circuit crashes
    with IndexError during weights assignment.
    """
    circuit = {
        'inputs': ['x_0', 'x_0'],
        'gates': {'g_0': ('AND', ['x_0', 'x_0'])},
        'outputs': {'y_0': 'g_0'}
    }
    with pytest.raises(ValueError, match="Duplicate input names are not allowed"):
        compile_mlp_from_circuit(circuit, alphabet_size=2)

def test_evaluate_circuit_list_outputs_boolean_coercion():
    """
    Test evaluate_circuit list outputs boolean coercion bug:
    When outputs is a list, evaluate_circuit uses boolean mode even if inputs
    are characters, returning [True] instead of ['a'].
    """
    circuit = {
        'inputs': ['x_0'],
        'gates': {'g_0': ('IDENTITY', ['x_0'])},
        'outputs': ['g_0']
    }
    # With list outputs, it now correctly returns ['a'] (character mode)
    res_list = evaluate_circuit(circuit, ['a'])
    assert res_list == ['a']

    # With dict outputs, it returns ['a'] (character mode)
    circuit_dict = {
        'inputs': ['x_0'],
        'gates': {'g_0': ('IDENTITY', ['x_0'])},
        'outputs': {'y_0': 'g_0'}
    }
    res_dict = evaluate_circuit(circuit_dict, ['a'])
    assert res_dict == ['a']

def test_solve_cegis_index_error_reproduction():
    """
    Test solve_cegis IndexError reproduction:
    When Z3 CEGIS needs to resolve a gate-to-gate connection (e.g. depth >= 2),
    it raises IndexError: list index out of range when building gate names.
    """
    x_0, x_1 = sympy.symbols('x_0 x_1')
    temp = sympy.Symbol('r_0')
    replacements = [(temp, sympy.Not(x_0))]
    reduced = [sympy.And(temp, x_1)]
    
    res = solve_cegis(
        d=2,
        K=2,
        input_names=['x_0', 'x_1'],
        y_vars=['y_0'],
        replacements=replacements,
        reduced=reduced,
        seq_len=2,
        start_time=time.time(),
        timeout=10.0
    )
    assert isinstance(res, dict)
    assert "gates" in res
    assert "outputs" in res

def test_synthesize_circuit_fractional_routing_timeout():
    """
    Test synthesize_circuit with fractional routing:
    Verify it returns 'TIMEOUT' under a tight timeout, preventing infinite hangs.
    """
    R = [[0.5, 0.5]]
    res = synthesize_circuit(
        seq_len=2,
        num_outputs=1,
        alphabet=['a', 'b'],
        routing_matrix=R,
        timeout=0.1
    )
    assert res == "TIMEOUT"
