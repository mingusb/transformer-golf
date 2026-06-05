import sys
import random
from collections import defaultdict, Counter
import argparse

def build_markov_model(text, order=3):
    """
    Builds an n-gram Markov model from the given text.
    'order' is the number of characters representing the current state.
    """
    model = defaultdict(Counter)
    # Slide a window of length 'order' across the text
    for i in range(len(text) - order):
        state = text[i:i+order]
        next_char = text[i+order]
        model[state][next_char] += 1
    return model

def generate_text(model, order, length=200, seed=None):
    """
    Generates text using the trained Markov model.
    """
    if not model:
        return ""
        
    if seed is None or seed not in model:
        seed = random.choice(list(model.keys()))
    
    current_state = seed
    output = list(current_state)
    
    for _ in range(length - order):
        if current_state not in model:
            # If we hit a dead end, pick a new random state
            current_state = random.choice(list(model.keys()))
            
        choices = list(model[current_state].keys())
        weights = list(model[current_state].values())
        
        next_char = random.choices(choices, weights=weights)[0]
        output.append(next_char)
        
        # Update state to the last 'order' characters
        current_state = "".join(output[-order:])
        
    return "".join(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shannon's Original Markovian Text Generator (1948)")
    parser.add_argument("--corpus", type=str, default="../regex_corpus.txt", help="Path to corpus file")
    parser.add_argument("--order", type=int, default=4, help="Order of the Markov chain (n-gram size)")
    parser.add_argument("--length", type=int, default=500, help="Length of generated text")
    args = parser.parse_args()

    try:
        with open(args.corpus, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Corpus file '{args.corpus}' not found.")
        sys.exit(1)

    print(f"--- Training {args.order}-Order Markov Model on {len(text)} characters ---")
    model = build_markov_model(text, order=args.order)
    print(f"Model built with {len(model)} unique states.")
    
    print("\n--- Generating Text (Claude Shannon's N-Gram Approximation) ---")
    generated = generate_text(model, order=args.order, length=args.length)
    print(generated)
