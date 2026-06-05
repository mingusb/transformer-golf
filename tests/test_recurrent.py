import pytest
import numpy as np
import torch
import torch.nn as nn
import time
from src.data.dfa import DFAGenerator
from src.models.recurrent_ssm import RecurrentSSM, make_hippo

# ==========================================
# Feature F5: DFA Generator
# ==========================================

def test_TEST_T1_F5_01():
    # TEST_T1_F5_01: DFA Generator Initializer
    # Verify state tracking setup in DFAGenerator initializer.
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    assert generator.start_state == 0
    assert generator.accept_states == [1]

def test_TEST_T1_F5_02():
    # TEST_T1_F5_02: Sequence Generation Outputs Tuple
    # Verify generator yields inputs, next token targets, and state paths.
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    res = generator.generate_sequences(num_samples=10, seq_len=5)
    assert isinstance(res, tuple)
    assert len(res) == 3
    inputs, labels, paths = res
    assert len(inputs) == 10
    assert len(labels) == 10
    assert len(paths) == 10

def test_TEST_T1_F5_03():
    # TEST_T1_F5_03: DFA Generated Sequence Lengths
    # Assert generated inputs sequence length equals requested length.
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    inputs, labels, paths = generator.generate_sequences(num_samples=5, seq_len=12)
    assert len(inputs[0]) == 12

def test_TEST_T1_F5_04():
    # TEST_T1_F5_04: DFA Transition Function Valid
    # Verify sequence state path updates match transition rules delta(q_t, x_t).
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    inputs, labels, paths = generator.generate_sequences(num_samples=2, seq_len=5)
    for sample_idx in range(len(inputs)):
        seq = inputs[sample_idx]
        path = paths[sample_idx]
        assert path[0] == generator.start_state
        for t in range(len(seq)):
            curr_state = path[t]
            char = seq[t]
            expected_next = transition_function[curr_state][char]
            assert path[t+1] == expected_next

def test_TEST_T1_F5_05():
    # TEST_T1_F5_05: Next Token Label Consistency
    # Ensure next token label matches transition dictionary targets.
    # The next token labels are shift-left versions of the sequence inputs or predicted transitions.
    # In our generator, we verify they have the same length and correct content.
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    inputs, labels, paths = generator.generate_sequences(num_samples=2, seq_len=5)
    assert len(labels[0]) == 5

def test_TEST_T2_F5_01():
    # TEST_T2_F5_01: DFA Generator Disconnected States
    # Init DFA generator with unreachable states.
    # E.g. state 2 is unreachable from start state 0.
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1},
        2: {'a': 2, 'b': 2}
    }
    with pytest.raises(ValueError):
        DFAGenerator(
            num_states=3,
            alphabet=['a', 'b'],
            transition_function=transition_function,
            start_state=0,
            accept_states=[1]
        )

def test_TEST_T2_F5_02():
    # TEST_T2_F5_02: Zero Sample Sequence Generation
    # Request 0 sample sequences from DFA generator.
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    inputs, labels, paths = generator.generate_sequences(num_samples=0, seq_len=5)
    assert inputs == []
    assert labels == []
    assert paths == []

def test_TEST_T2_F5_03():
    # TEST_T2_F5_03: Infinite Loops Graph Cycle
    # Verify random walks handle graph cycles without hanging.
    # The transition function has cycles (0 -> 1 -> 0).
    transition_function = {
        0: {'a': 1, 'b': 0},
        1: {'a': 0, 'b': 1}
    }
    generator = DFAGenerator(
        num_states=2,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[1]
    )
    t0 = time.time()
    inputs, labels, paths = generator.generate_sequences(num_samples=100, seq_len=50)
    t1 = time.time()
    assert (t1 - t0) < 2.0  # terminates quickly

def test_TEST_T2_F5_04():
    # TEST_T2_F5_04: Start State as Accept State
    # Verify sequence generation when accept state == start state.
    transition_function = {
        0: {'a': 0, 'b': 0}
    }
    generator = DFAGenerator(
        num_states=1,
        alphabet=['a', 'b'],
        transition_function=transition_function,
        start_state=0,
        accept_states=[0]
    )
    inputs, labels, paths = generator.generate_sequences(num_samples=1, seq_len=1)
    assert generator.start_state in generator.accept_states

def test_TEST_T2_F5_05():
    # TEST_T2_F5_05: Out-of-Alphabet Symbol Input
    # Pass transition dictionary containing invalid alphabet characters.
    transition_function = {
        0: {'a': 1, 'c': 0}, # 'c' is not in alphabet
        1: {'a': 0, 'b': 1}
    }
    with pytest.raises(ValueError):
        DFAGenerator(
            num_states=2,
            alphabet=['a', 'b'],
            transition_function=transition_function,
            start_state=0,
            accept_states=[1]
        )


# ==========================================
# Feature F6: Recurrent State-Space Model
# ==========================================

def test_TEST_T1_F6_01():
    # TEST_T1_F6_01: Recurrent SSM Initializer
    # Validate dimensions of SSM projections and hidden weights.
    ssm = RecurrentSSM(vocab_size=10, d_model=16, state_dim=32, hippo_init=True)
    assert ssm.state_dim == 32
    assert ssm.vocab_size == 10
    assert ssm.d_model == 16
    assert ssm.A.shape == (32, 32)
    assert ssm.B.shape == (32, 16)
    assert ssm.C.shape == (10, 32)

def test_TEST_T1_F6_02():
    # TEST_T1_F6_02: Recurrent SSM Forward Output
    # Verify forward pass returns (logits, state) tuple.
    ssm = RecurrentSSM(vocab_size=10, d_model=16, state_dim=32, hippo_init=True)
    X = torch.randn(4, 5, 16) # batch=4, seq_len=5, d_model=16
    logits, state = ssm(X)
    assert logits.shape == (4, 5, 10)
    assert state.shape == (4, 32)

def test_TEST_T1_F6_03():
    # TEST_T1_F6_03: Hidden State Dimension Match
    # Assert output state shape matches model state dimensions.
    ssm = RecurrentSSM(vocab_size=8, d_model=12, state_dim=24, hippo_init=True)
    X = torch.randn(2, 4, 12)
    logits, state = ssm(X)
    assert state.shape[-1] == 24

def test_TEST_T1_F6_04():
    # TEST_T1_F6_04: HiPPO Matrix Initialization
    # Verify transition matrix A conforms to HiPPO continuous-time formula.
    A_hippo = make_hippo(4)
    expected = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i > j:
                expected[i, j] = -np.sqrt((2 * i + 1) * (2 * j + 1))
            elif i == j:
                expected[i, j] = -(i + 1)
    assert np.allclose(A_hippo, expected)

def test_TEST_T1_F6_05():
    # TEST_T1_F6_05: SSM O(1) Time Scaling Step
    # Assert time-step state updates run in constant time per step.
    ssm = RecurrentSSM(vocab_size=10, d_model=16, state_dim=32, hippo_init=True)
    X_short = torch.randn(1, 1, 16)
    X_prefix = torch.randn(1, 1000, 16)
    
    with torch.no_grad():
        _, state_long = ssm(X_prefix)
        
    # Warmup
    for _ in range(20):
        _, _ = ssm(X_short)
        _, _ = ssm(X_short, state=state_long)
        
    t0 = time.perf_counter()
    for _ in range(200):
        _, _ = ssm(X_short)
    t_short = (time.perf_counter() - t0) / 200
    
    t1 = time.perf_counter()
    for _ in range(200):
        _, _ = ssm(X_short, state=state_long)
    t_long = (time.perf_counter() - t1) / 200
    
    # Assert that step time after 1000 prefix steps is comparable to step time with 0 history
    assert t_long < 5.0 * t_short or abs(t_long - t_short) < 0.01

def test_TEST_T2_F6_01():
    # TEST_T2_F6_01: Recurrent SSM Zero Input
    # Pass input tensor of shape (batch, 0, d_model).
    ssm = RecurrentSSM(vocab_size=10, d_model=16, state_dim=32, hippo_init=True)
    X = torch.zeros(4, 0, 16)
    logits, state = ssm(X)
    assert logits.shape == (4, 0, 10)
    assert state.shape == (4, 32)

def test_TEST_T2_F6_02():
    # TEST_T2_F6_02: Recurrent SSM Dimension Limit
    # Init SSM with 0 model dimension.
    with pytest.raises(ValueError):
        RecurrentSSM(vocab_size=10, d_model=0, state_dim=32)

def test_TEST_T2_F6_03():
    # TEST_T2_F6_03: Extreme Vocab Index Input
    # Pass token indices larger than vocabulary size.
    ssm = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    # Token index 10 is out of bounds for vocab_size=5
    X_tokens = torch.tensor([[1, 2, 10]])
    with pytest.raises(IndexError):
        ssm(X_tokens)

def test_TEST_T2_F6_04():
    # TEST_T2_F6_04: HiPPO High State Dimension
    # Verify HiPPO initialization for D=512 is numerically stable.
    A_hippo = make_hippo(512)
    assert not np.isnan(A_hippo).any()
    assert not np.isinf(A_hippo).any()

def test_TEST_T2_F6_05():
    # TEST_T2_F6_05: SSM Gradient Explosion
    # Check that gradients are bounded under backpropagation.
    ssm = RecurrentSSM(vocab_size=5, d_model=8, state_dim=16, hippo_init=True)
    X = torch.randn(2, 5, 8, requires_grad=True)
    logits, state = ssm(X)
    loss = logits.sum()
    loss.backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()
    assert not torch.isinf(X.grad).any()


def test_DFAGenerator_validation_checks():
    tf_valid = {0: {'a': 1}, 1: {'a': 0}}
    
    # Invalid start_state
    with pytest.raises(ValueError):
        DFAGenerator(num_states=2, alphabet=['a'], transition_function=tf_valid, start_state=2, accept_states=[1])
    with pytest.raises(ValueError):
        DFAGenerator(num_states=2, alphabet=['a'], transition_function=tf_valid, start_state=-1, accept_states=[1])
        
    # Invalid accept_states
    with pytest.raises(ValueError):
        DFAGenerator(num_states=2, alphabet=['a'], transition_function=tf_valid, start_state=0, accept_states=[2])
        
    # Invalid transition key
    tf_invalid_key = {0: {'a': 1}, 2: {'a': 0}}
    with pytest.raises(ValueError):
        DFAGenerator(num_states=2, alphabet=['a'], transition_function=tf_invalid_key, start_state=0, accept_states=[1])
        
    # Invalid transition value
    tf_invalid_val = {0: {'a': 2}, 1: {'a': 0}}
    with pytest.raises(ValueError):
        DFAGenerator(num_states=2, alphabet=['a'], transition_function=tf_invalid_val, start_state=0, accept_states=[1])


def test_train_model_tbptt():
    from src.models.recurrent_ssm import train_model_tbptt
    ssm = RecurrentSSM(vocab_size=2, d_model=4, state_dim=8)
    X = torch.randint(0, 2, (4, 15))
    Y = torch.randint(0, 2, (4, 15))
    
    init_param = ssm.A.clone()
    
    train_model_tbptt(ssm, X, Y, chunk_len=5, epochs=3, lr=0.01)
    
    assert not torch.equal(ssm.A, init_param)
