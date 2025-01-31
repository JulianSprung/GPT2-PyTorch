import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, head_size: int, context_size: int, dropout: float, n_embed: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Create a larger tril matrix to handle different sequence lengths
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        # Ensure we only use the appropriate part of the tril matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_size: int,
        context_size: int,
        n_embed: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(head_size, context_size, dropout, n_embed)
                for _ in range(n_heads)
            ]
        )
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embed: int,
        context_size: int,
        n_heads: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            n_heads, head_size, context_size, n_embed, dropout
        )
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        head_size: int,
        context_size: int,
        n_layers,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.context_size = context_size
        self.pos_emb = nn.Embedding(context_size, n_embed)
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(n_embed, context_size, n_heads, head_size, dropout)
                for _ in range(n_layers)
            ]
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
        # Generate one token at a time
        for _ in range(max_new_tokens):
            # Crop the context to the last context size tokens if it's too long
            ctx = self.context_size
            context = tokens[:, -ctx:] if tokens.size(1) > ctx else tokens
            logits, loss = self(context)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tokens = torch.cat((tokens, next_token), dim=1)  # (B, T+1)
        return tokens
