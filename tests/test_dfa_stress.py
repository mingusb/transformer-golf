import time
import random
import numpy as np
import pytest
from src.data.dfa import DFAGenerator

def generate_large_cyclic_dfa(num_states=200, alphabet_size=100):
    # Alphabet of size alphabet_size
    alphabet = [f"sym_{i}" for i in range(alphabet_size)]
    
    # Initialize transition function
    transition_function = {}
    
    # To guarantee reachability of all states, we build a cycle: i -> (i+1)%num_states
    # on the first symbol.
    cycle_symbol = alphabet[0]
    
    for state in range(num_states):
        transition_function[state] = {}
        # Cycle transition
        next_state_cycle = (state + 1) % num_states
        transition_function[state][cycle_symbol] = next_state_cycle
        
        # Other symbols get random transitions
        for symbol in alphabet[1:]:
            transition_function[state][symbol] = random.randint(0, num_states - 1)
            
    start_state = 0
    accept_states = [num_states - 1]
    
    return num_states, alphabet, transition_function, start_state, accept_states

# 1. Stress tests for extreme inputs
def test_dfa_extreme_stress():
    print("\n[STRESS TEST] Generating large cyclic DFA (states=200, alphabet=100)...")
    num_states, alphabet, tf, start, accept = generate_large_cyclic_dfa(num_states=200, alphabet_size=100)
    
    # Measure generator initialization time
    t_start = time.perf_counter()
    generator = DFAGenerator(
        num_states=num_states,
        alphabet=alphabet,
        transition_function=tf,
        start_state=start,
        accept_states=accept
    )
    t_init = time.perf_counter() - t_start
    print(f"[STRESS TEST] Initialization took {t_init:.4f} seconds")
    assert t_init < 1.0, f"Initialization took too long: {t_init:.2f}s"
    
    # Test a single walk of length 1000
    t_single_start = time.perf_counter()
    inputs, labels, paths = generator.generate_sequences(num_samples=1, seq_len=1000)
    t_single = time.perf_counter() - t_single_start
    print(f"[STRESS TEST] Generating 1 sample of length 1000 took {t_single:.4f} seconds")
    assert t_single < 1.0, f"Single walk of length 1000 took too long: {t_single:.2f}s"
    
    # Test 10 walks of length 1000
    t_ten_start = time.perf_counter()
    inputs, labels, paths = generator.generate_sequences(num_samples=10, seq_len=1000)
    t_ten = time.perf_counter() - t_ten_start
    print(f"[STRESS TEST] Generating 10 samples of length 1000 took {t_ten:.4f} seconds")
    assert t_ten < 1.0, f"10 walks of length 1000 took too long: {t_ten:.2f}s"
    
    # Verify correctness of walks
    for i in range(len(inputs)):
        seq = inputs[i]
        path = paths[i]
        assert path[0] == start
        curr = start
        for step, sym in enumerate(seq):
            expected_next = tf[curr][sym]
            assert path[step + 1] == expected_next, f"Mismatch at step {step}: expected {expected_next}, got {path[step + 1]}"
            curr = expected_next

# 2. Validation tests for invalid inputs
def test_dfa_invalid_inputs():
    print("\n[VALIDATION TEST] Testing invalid inputs...")
    alphabet = ['a', 'b']
    
    # Base valid parameters
    valid_tf = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    
    # 2.1 Out-of-bounds start state
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=valid_tf, start_state=2, accept_states=[1])
    assert "start_state" in str(excinfo.value)
    
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=valid_tf, start_state=-1, accept_states=[1])
    assert "start_state" in str(excinfo.value)
    
    # 2.2 Out-of-bounds accept state
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=valid_tf, start_state=0, accept_states=[2])
    assert "accept_state" in str(excinfo.value)
    
    # 2.3 Invalid transition key (state index out of range)
    invalid_key_tf = {
        0: {'a': 1, 'b': 0},
        2: {'a': 0, 'b': 1}
    }
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=invalid_key_tf, start_state=0, accept_states=[1])
    assert "transition_function key" in str(excinfo.value)
    
    # 2.4 Invalid transition value (target state out of range)
    invalid_val_tf = {
        0: {'a': 2, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=invalid_val_tf, start_state=0, accept_states=[1])
    assert "transition_function value" in str(excinfo.value)
    
    # 2.5 Disconnected/unreachable state
    disconnected_tf = {
        0: {'a': 0, 'b': 0},
        1: {'a': 1, 'b': 1}
    }
    # State 1 is unreachable from start state 0
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=disconnected_tf, start_state=0, accept_states=[0])
    assert "Unreachable states detected" in str(excinfo.value)

    # 2.6 Out of alphabet transition character
    out_of_alph_tf = {
        0: {'a': 1, 'c': 0},
        1: {'a': 0, 'b': 1}
    }
    with pytest.raises(ValueError) as excinfo:
        DFAGenerator(num_states=2, alphabet=alphabet, transition_function=out_of_alph_tf, start_state=0, accept_states=[1])
    assert "not in alphabet" in str(excinfo.value)

if __name__ == "__main__":
    test_dfa_extreme_stress()
    test_dfa_invalid_inputs()
    print("\nAll stress tests passed successfully!")
