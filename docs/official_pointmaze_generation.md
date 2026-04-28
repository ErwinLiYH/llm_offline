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
`local_varient_gen.py`.

Local PointMaze generation uses official core code:

- `scripts/pointmaze/controller.py::WaypointController`
- `scripts/pointmaze/maze_solver.py::QIteration`
- `scripts/pointmaze/create_pointmaze_dataset.py::PointMazeStepDataCallback`

Generate a local variant:

```bash
micromamba run -n d4rl_datagen python local_varient_gen.py \
  --variants local-layout-01 \
  --num-workers 2 \
  --target-episodes 20 \
  --overwrite \
  --seed 42
```

Generated datasets live under `local_datasets/` and are ignored by Git.
