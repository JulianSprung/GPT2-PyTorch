import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
CONTEXT_SIZE = 8
LEARNING_RATE = 1e-3

with open("data/input.txt", "r") as f:
    text = f.read()

vocab = sorted(set(text))
vocab_size = len(vocab)

# Building a "tokenizer"
char_to_ind = {v: i for i, v in enumerate(vocab)}
ind_to_char = {i: v for i, v in enumerate(vocab)}


def encode(text: str) -> list[int]:
    return [char_to_ind[c] for c in text]


def decode(tokens: list[int]) -> str:
    return "".join([ind_to_char[i] for i in tokens])


tokens = encode(text)

tokens = torch.tensor(tokens)
train_split = int(0.9 * len(tokens))
train_tokens = tokens[:train_split]
val_tokens = tokens[train_split:]


def get_batch(split: str):
    data = train_tokens if split == "train" else val_tokens
    ix = torch.randint(len(data) - BATCH_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_SIZE + 1] for i in ix])
    return x, y


class BigrammModel(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, tokens, targets=None):
        token_emb = self.embedding(tokens)
        logits = self.lm_head(token_emb)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, tokens, max_new_tokens):

        for _ in range(max_new_tokens):
            logits, loss = self(tokens)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


m = BigrammModel(vocab_size, 32)
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for steps in range(10000):
    x, y = get_batch("train")
    y_hat, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 1000 == 0:
        print(f"loss at step {steps} is {loss.item()}")

print("Final loss:", loss.item())

# validation loss
x, y = get_batch("val")
y_hat, loss = m(x, y)
print(f"validation loss: {loss.item()}")

print("Generating text...")
print(
    decode(
        m.generate(tokens=torch.zeros([1, 1], dtype=torch.long), max_new_tokens=300)[
            0
        ].tolist()
    )
)

print("End of file")
