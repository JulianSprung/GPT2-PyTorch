import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPT2Config:
    vocab_size: int
    context_size: int
    n_layers: int
    n_heads: int
    n_embed: int
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_heads == 0
        # c_attn creates k,q,v as a batch
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 3)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        # Using "bias" instead of "tril" to match the OAI/HF GPT-2 implementation wording
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
            ),
        )

    def forward(self, x: Tensor):
        BS, SL, ED = x.size()  # batch size, sequence length, embedding dimension
        qkv: Tensor = self.c_attn(x)  # (BS, SL, 3*ED)
        q, k, v = qkv.split(split_size=self.config.n_embed, dim=2)  # (BS, SL, ED)
        k: Tensor = k.view(
            BS, SL, self.config.n_heads, ED // self.config.n_heads
        )  # (BS, SL, n_heads, head_size)
        k = k.transpose(1, 2)  # (BS, n_heads, SL, head_size)
        q: Tensor = q.view(
            BS, SL, self.config.n_heads, ED // self.config.n_heads
        )  # (BS, SL, n_heads, head_size)
        q = q.transpose(1, 2)  # (BS, n_heads, SL, head_size)
        v: Tensor = v.view(
            BS, SL, self.config.n_heads, ED // self.config.n_heads
        )  # (BS, SL, n_heads, head_size)
        v = v.transpose(1, 2)  # (BS, n_heads, SL, head_size)

        att: Tensor = (q @ k.transpose(-2, -1)) * k.size(
            -1
        ) ** -0.5  # (..., SL, head_size) @ (..., head_size, SL) -> (..., SL, SL)
        att = att.masked_fill(self.bias[:, :, :SL, :SL] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)  # (..., SL, SL)
        y = att @ v  # (..., SL, SL) @ (..., SL, head_size) -> (..., SL, head_size)
        # Basically concatenate the heads into one tensor
        y = (
            y.transpose(1, 2).contiguous().view(BS, SL, ED)
        )  # (..., n_heads, SL, head_size) -> (..., SL, n_heads*head_size)

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.context_size, config.n_embed),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                "ln_f": nn.LayerNorm(config.n_embed),
            }
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, x, targets=None):
        BS, SL = x.shape
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(torch.arange(0, SL, device=x.device))
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layers=24, n_heads=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layers=36, n_heads=20, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_heads=25, n_embed=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["context_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
