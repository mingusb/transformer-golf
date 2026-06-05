import numpy as np
import warnings

class DFAGenerator:
    def __init__(self, num_states, alphabet, transition_function, start_state, accept_states):
        self.num_states = num_states
        self.alphabet = [str(a) for a in alphabet]
        self.transition_function = {}
        for state, trans in transition_function.items():
            self.transition_function[state] = {str(k): v for k, v in trans.items()}
        self.start_state = start_state
        self.accept_states = accept_states

        # Validate that start_state, accept_states, and transition_function states are within range(num_states)
        if not (0 <= start_state < num_states):
            raise ValueError(f"start_state {start_state} is not within range(num_states)")
        for state in accept_states:
            if not (0 <= state < num_states):
                raise ValueError(f"accept_state {state} is not within range(num_states)")
        for state, trans in self.transition_function.items():
            if not (0 <= state < num_states):
                raise ValueError(f"transition_function key {state} is not within range(num_states)")
            for next_state in trans.values():
                if not (0 <= next_state < num_states):
                    raise ValueError(f"transition_function value {next_state} is not within range(num_states)")

        # Check that transition function inner keys represent valid alphabet characters
        for state, trans in self.transition_function.items():
            for char in trans.keys():
                if char not in self.alphabet:
                    raise ValueError(f"Character {char} not in alphabet")

        # Check reachability of states
        visited = set()
        queue = [start_state]
        while queue:
            s = queue.pop(0)
            if s not in visited:
                visited.add(s)
                if s in self.transition_function:
                    for next_s in self.transition_function[s].values():
                        queue.append(next_s)

        all_states = set(range(num_states))
        unreachable = all_states - visited
        if unreachable:
            warnings.warn(f"Unreachable states detected: {unreachable}")
            raise ValueError(f"Unreachable states detected: {unreachable}")

    def generate_sequences(self, num_samples, seq_len):
        if num_samples < 0 or seq_len < 0:
            raise ValueError("Sample count and sequence length must be non-negative")
        if num_samples == 0 or seq_len == 0:
            return [], [], []

        input_sequences = []
        next_token_labels = []
        state_paths = []

        for _ in range(num_samples):
            seq = []
            states = [self.start_state]
            current_state = self.start_state

            for _ in range(seq_len):
                char = str(np.random.choice(self.alphabet))
                seq.append(char)

                if current_state in self.transition_function:
                    next_state = self.transition_function[current_state].get(char)
                else:
                    next_state = None

                if next_state is None:
                    next_state = self.start_state

                states.append(next_state)
                current_state = next_state

            # For next-token labels, shift left and choose a random alphabet char for the last step
            labels = seq[1:] + [str(np.random.choice(self.alphabet))]

            input_sequences.append(seq)
            next_token_labels.append(labels)
            state_paths.append(states)

        return input_sequences, next_token_labels, state_paths
