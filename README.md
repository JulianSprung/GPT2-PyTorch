# GPT2-PyTorch
Pytorch implementation of a transformer and evolving into GPT-2 

## Getting started
Install all dependencies.
```bash
poetry install
```
Install the non-dev (e.g. without Jupyter) only
```bash
poetry install --only main
```

## Data
The tiny shakeaspeare can be fetched via
```bash
wget -O data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
or
```bash
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Training
```bash
python -m src.transformer.train
```
