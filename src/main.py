import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
BATCH_SIZE = 32
CONTEXT_SIZE = 8
LEARNING_RATE = 1e-3
N_HEADS = 4
HEAD_SIZE = 16
N_EMBED = 32
DROPOUT = 0.0
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
N_LAYERS = 4


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

tokens = torch.tensor(tokens).to(DEVICE)
train_split = int(0.9 * len(tokens))
train_tokens = tokens[:train_split]
val_tokens = tokens[train_split:]


def get_batch(split: str):
    data = train_tokens if split == "train" else val_tokens
    ix = torch.randint(len(data) - BATCH_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + CONTEXT_SIZE] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + CONTEXT_SIZE + 1] for i in ix]).to(DEVICE)
    return x, y


class AttentionHead(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()

        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.mha = MultiHeadAttention(N_HEADS, HEAD_SIZE)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.mha(x)
        x = x + self.ffwd(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.pos_emb = nn.Embedding(CONTEXT_SIZE, n_embed)
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embed) for _ in range(N_LAYERS)]
        )
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        token_emb = self.embedding(tokens)
        pos = torch.arange(T, device=tokens.device)
        x = token_emb + self.pos_emb(pos)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)
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


m = Transformer(vocab_size, N_EMBED)
m = m.to(DEVICE)
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
x_val, y_val = get_batch("val")
y_hat_val, loss_val = m(x_val, y_val)
print(f"validation loss: {loss_val.item()}")

print("Generating text...")
print(
    decode(
        m.generate(
            tokens=torch.zeros([1, 1], dtype=torch.long, device=DEVICE),
            max_new_tokens=300,
        )[0].tolist()
    )
)

print("End of file")
