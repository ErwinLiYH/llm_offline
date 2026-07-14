# Official PointMaze Dataset Generation

This project vendors Farama's official dataset generation scripts as a Git
submodule at:

```bash
third_party/minari-dataset-generation-scripts
```

Initialize it after cloning this repo:

```bash
git submodule update --init --recursive
```

Create the dedicated data-generation environment:

```bash
micromamba env create -f dataGen_env.yaml
```

Update it intentionally:

```bash
git -C third_party/minari-dataset-generation-scripts fetch origin
git -C third_party/minari-dataset-generation-scripts checkout <official-commit>
git add third_party/minari-dataset-generation-scripts .gitmodules
```

The parent repository records the exact official commit. Do not edit files
inside the submodule for project-specific behavior; keep compatibility logic in
`local_pointmaze_gen.py`.

Local PointMaze generation uses official core code:

- `scripts/pointmaze/controller.py::WaypointController`
- `scripts/pointmaze/maze_solver.py::QIteration`

The Minari step callback lives in `local_pointmaze_gen.py` so local generation
does not import the official `create_pointmaze_dataset.py` check path or its
extra diagnostics dependencies.

Generate a local variant:

```bash
micromamba run -n d4rl_datagen python local_pointmaze_gen.py \
  --variants local-layout-01 \
  --num-workers 2 \
  --target-episodes 20 \
  --reward-type dense \
  --overwrite \
  --seed 42
```

Generate local data with a post-success hold segment:

```bash
micromamba run -n d4rl_datagen python local_pointmaze_gen.py \
  --variants local-layout-07 \
  --num-workers 4 \
  --target-episodes 1000 \
  --post-success-hold-steps 100 \
  --post-success-hold-noise-std 0.0 \
  --overwrite \
  --seed 42
```

`--post-success-hold-steps` keeps recording after the first goal reach in each
episode. During this phase the generator uses a deterministic PD hold action by
default, so the data teaches the policy to stay near the fixed goal. Use
`--post-success-hold-noise-std` to add optional Gaussian action noise only
during the hold phase. Use `--overwrite` when enabling hold data for an
existing local dataset to avoid mixing old goal-arrival-only episodes with hold
episodes.

`--reward-type` accepts `sparse` or `dense` and defaults to the registered
variant reward type. The default dataset path remains unchanged. Choosing the
alternate reward writes a separate dataset, for example
`pointmaze-local-layout-01-dense-v0`, so sparse and dense rewards cannot be
mixed by append. `generation_summary.json` records the effective reward type.

Generated datasets live under `local_datasets/` and are ignored by Git.
