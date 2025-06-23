# split_conll.py

import random

input_file = "C:\\Users\\HP\\EthioMart-NER\\amharic_ner.conll"
train_file = "C:\\Users\\HP\\EthioMart-NER\\train.conll"
valid_file = "C:\\Users\\HP\\EthioMart-NER\\valid.conll"

# Read the whole file and split by sentences (empty line separates sentences)
with open(input_file, encoding="utf-8") as f:
    content = f.read()

sentences = content.strip().split("\n\n")
print(f"Total sentences: {len(sentences)}")

random.shuffle(sentences)

# Split 80% train, 20% validation
split_index = int(0.8 * len(sentences))
train_sents = sentences[:split_index]
valid_sents = sentences[split_index:]

# Write train file
with open(train_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_sents) + "\n")

# Write valid file
with open(valid_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(valid_sents) + "\n")

print(f"Train sentences: {len(train_sents)}")
print(f"Validation sentences: {len(valid_sents)}")
print("Files created: train.conll, valid.conll")
