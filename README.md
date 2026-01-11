# Mini R1

This repo contains a mini implementation of the DeepSeek R1 pipeline intended to run on a single 8xH100 box.


**todos**
- [ ] add inference engine support
- [ ] enable multinode distributed w/ fsdp2
- [ ] reference policy updates
- [ ] use r1 hyperparams
- [ ] padding-free training
- [ ] ffd packing
- [ ] eval harness
- [ ] proper math reasoning tasks
- [ ] wandb
- [ ] consolidate trainer


## Install:

Create a new Python environment and activate it:

```bash
uv venv --python=3.12 
uv sync  --all-extras  # includes flash-attention 
```

## Usage

To use mini-grpo, you can follow these steps:

< tbd >

### Create Training Data

Generate data like this:

```bash
python cli.py generate-data  # additional args here
```


### Train the model with GRPO

You can launch GRPO training with the following command:

```bash
python cli.py orchestrator --num-epochs 10

```
