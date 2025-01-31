import torch

from .transformer import Transformer

# Hyperparameters
BATCH_SIZE = 8
CONTEXT_SIZE = 32
LEARNING_RATE = 1e-3
N_HEADS = 8
HEAD_SIZE = 16
N_EMBED = 32
DROPOUT = 0.0
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
N_LAYERS = 4

with open("data/tiny-shakespeare.txt", "r") as f:
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
    # Ensure we have enough context for each index
    ix = torch.randint(0, len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + CONTEXT_SIZE] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + CONTEXT_SIZE + 1] for i in ix]).to(DEVICE)
    return x, y


m = Transformer(
    vocab_size, N_EMBED, N_HEADS, HEAD_SIZE, CONTEXT_SIZE, N_LAYERS, DROPOUT
)
m = m.to(DEVICE)

print("Parameters:", sum(p.numel() for p in m.parameters()))

optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

print("# Generating text (before training)...")
print(
    decode(
        m.generate(
            tokens=torch.zeros([1, 1], dtype=torch.long, device=DEVICE),
            max_new_tokens=300,
        )[0].tolist()
    )
)
print("# Training...")
for steps in range(2000):
    x, y = get_batch("train")
    y_hat, loss = m(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 1000 == 0:
        x_val, y_val = get_batch("val")
        y_hat_val, loss_val = m(x_val, y_val)
        print(
            f"  loss at step {steps} is {loss.item()}. Validation loss: {loss_val.item()}"
        )
x_val, y_val = get_batch("val")
y_hat_val, loss_val = m(x_val, y_val)
print("  final loss:", loss.item(), "validation loss:", loss_val.item())


print("# Generating text (after training)...")
print(
    decode(
        m.generate(
            tokens=torch.zeros([1, 1], dtype=torch.long, device=DEVICE),
            max_new_tokens=300,
        )[0].tolist()
    )
)

print("#End of file")
