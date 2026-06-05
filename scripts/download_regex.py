import os
from datasets import load_dataset

print("Downloading regex dataset from Hugging Face...")
# The dataset has columns like 'pattern', 'description', etc.
dataset = load_dataset("innovatorved/regex_dataset", split="train")

print(f"Loaded {len(dataset)} regular expressions. Writing to regex_corpus.txt...")

with open("regex_corpus.txt", "w", encoding="utf-8") as f:
    for row in dataset:
        pattern = row.get("regex")
        if pattern:
            f.write(pattern + "\n")

print("Done! Corpus size:", os.path.getsize("regex_corpus.txt"), "bytes")
