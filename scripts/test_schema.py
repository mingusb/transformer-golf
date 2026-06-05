import os
from datasets import load_dataset

dataset = load_dataset("innovatorved/regex_dataset", split="train")
print(dataset[0])
