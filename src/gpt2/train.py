import torch

from .gpt2 import GPT2

model = GPT2.from_pretrained("gpt2")

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, I am a language model, "
tokens = tokenizer.encode(text)
x = torch.tensor(tokens).unsqueeze(0)

print(tokenizer.decode(x[0].tolist()))

# lets manually encode the text and sample from the model with a topk = 50 multinomial to select the next token etc in a while loop
while x.shape[1] < 100:
    with torch.no_grad():
        logits, _ = model(x)
    next_token = torch.multinomial(
        torch.softmax(logits[:, -1], dim=-1), num_samples=1
    )  # [B, 1]
    x = torch.cat([x, next_token], dim=1)  # [B, T+1]
    # Wrap the single token in a list for the decoder
    print(tokenizer.decode([next_token[0].item()]))

print("Generated text: ", tokenizer.decode(x[0].tolist()))
