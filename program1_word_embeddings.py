import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

corpus = """Natural language processing enables computers to understand human language.
Word embeddings map words into continuous vector space.
Neural networks can be used to learn these embeddings from context words in a document corpus."""
tokens = tokenize(corpus)
word_counts = Counter(tokens)
vocab = {word: i for i, word in enumerate(word_counts)}
vocab_size = len(vocab)

def generate_pairs(tokens, window_size=2):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i != j:
                pairs.append((center, tokens[j]))
    return pairs

pairs = generate_pairs(tokens)
training_data = [(vocab[center], vocab[context]) for center, context in pairs]

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, center_word):
        return self.output(self.embeddings(center_word))

model = Word2Vec(vocab_size, 50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(50):
    total_loss = 0.0
    for center, context in training_data:
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)
        optimizer.zero_grad()
        loss = criterion(model(center_tensor), context_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

embeddings = model.embeddings.weight.detach()
print("Word Embedding for 'language':\n", embeddings[vocab['language']])
print("Word Embedding for 'neural':\n", embeddings[vocab['neural']])
