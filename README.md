# Mini GRPO

This repo contains a simple implementation of the GRPO algorithm.

### Install:

```md
uv venv --python=3.12 
source .venv/bin/activate
uv pip install -e . && uv pip install flash-attn --no-build-isolation
```

### Usage

```md
python cli.py generate-data
python cli.py train
```
