# LLM Offline RL with CrossMaze

This project fine-tunes a large language model as a behavior-cloning policy for
Gymnasium Robotics **PointMaze** and **AntMaze** tasks. Offline trajectories are
converted into language prompts, and the model predicts either tokenized or
continuous actions. The repository also contains **CrossMaze**, a lightweight
environment package that adds structured maze layout and sensing data to normal
Gymnasium observations.

## Installation

The complete training stack expects a CUDA-capable GPU. Create the provided
environment and install the local CrossMaze package from the repository root:

```bash
micromamba create -n llm_offline -f environment.yaml
micromamba run -n llm_offline python -m pip install -e .
```

To use CrossMaze without the LLM training stack, only the editable installation
is needed; `pyproject.toml` installs its NumPy, Gymnasium, and
Gymnasium-Robotics dependencies.

```bash
python -m pip install -e .
```

## Using CrossMaze

`crossmaze.make` constructs a registered PointMaze or AntMaze evaluation
environment. It follows the standard Gymnasium API and adds a `crossmaze` entry
to every observation:

```python
import crossmaze

print(crossmaze.list_variants("pointmaze"))

env = crossmaze.make(
    "pointmaze",
    "umaze",
    mode="eval",
    config={
        "reward_type": "sparse",
        "wall_sensing_version": "v3",
    },
)

try:
    obs, info = env.reset(seed=0)
    maze = obs["crossmaze"]

    print(maze["maze_map"])          # static map: 0=open, 1=wall
    print(maze["dynamic_map"])       # 0=open, 1=wall, 2=current, 3=goal, 4=both
    print(maze["position_cell"])
    print(maze["goal_cell"])
    print(maze["neighbor_status"])   # [up, down, left, right]

    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
finally:
    env.close()
```

Neighbor status codes are `0=free`, `1=wall`, and `2=risk`. Supported families
are `pointmaze` and `antmaze`; call `crossmaze.list_variants(family)` to inspect
the registered variants.

CrossMaze can also build a deterministic procedural maze directly from a seed:

```python
env = crossmaze.make_seed_map(
    "pointmaze",
    map_seed=42,
    seed_map_spec={
        "size_mode": "fixed",
        "fixed_rows": 9,
        "fixed_cols": 9,
    },
    config={"reward_type": "sparse", "max_episode_steps": 300},
)
```

## Training an LLM policy

Training is configured entirely through YAML. The main settings are the base
model, environment family, training variants, prompt templates, action mode,
dataset selection, and optimization parameters. The example configs are:

- `config.yaml` for PointMaze
- `config.antmaze.yaml` for AntMaze

Remote D4RL datasets selected by these configs are downloaded through Minari.
Any selected local variant must already exist under `local_datasets/`, or under
the directory set by `local_dataset_root`.

Run training from the repository root:

```bash
# PointMaze
micromamba run -n llm_offline python train.py --config config.yaml

# AntMaze
micromamba run -n llm_offline python train.py --config config.antmaze.yaml
```

Later config files recursively override earlier ones, which makes small
experiment files easy to compose:

```bash
micromamba run -n llm_offline python train.py \
  --config config.yaml my_experiment.yaml
```

Use `--tokenize-only` to build and validate the dataset cache without loading
the model. Checkpoints are written below `checkpoint_root` (normally
`checkpoints/`), and the resolved run configuration is saved below
`exp_configs/`.

For single-node multi-GPU training:

```bash
micromamba run -n llm_offline torchrun --standalone --nproc_per_node=4 \
  train.py --config config.yaml --parallel_backend ddp
```

## Evaluating a checkpoint

Standalone evaluation performs environment rollouts and reports success and
return metrics. Set `model_path`, variants, episode count, and video options in
`eval.yaml` or `eval.antmaze.yaml`, or override the checkpoint on the command
line:

```bash
# PointMaze
micromamba run -n llm_offline python evaluate.py \
  --config eval.yaml --model_path checkpoints/.../final

# AntMaze
micromamba run -n llm_offline python evaluate.py \
  --config eval.antmaze.yaml --model_path checkpoints/.../final
```

A quoted trailing wildcard is accepted when it resolves to exactly one
checkpoint, for example `--model_path 'checkpoints/.../ep7*'`. Evaluation
outputs are written below `result_root` (`resultsV2/` in the provided configs).
Layered configs and DDP evaluation use the same conventions as training.

For official-style PointMaze normalized scoring, use the separate entry point:

```bash
micromamba run -n llm_offline python score.py --config score.yaml
```

More detailed design notes are available in [DESIGN.md](DESIGN.md), while
procedural-map data generation is documented in [docs/seed_map.md](docs/seed_map.md).
