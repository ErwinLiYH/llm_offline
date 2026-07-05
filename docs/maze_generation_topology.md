# Maze Generation and Topology Difficulty

This document describes the local maze-layout generation and static topology
difficulty tools used for custom AntMaze layouts. The same metric code can also
be applied to PointMaze maps, but the current inspection CLI is AntMaze-focused.

## Reference

The generator follows the design principle of design-centric maze generation:
define desired topological properties first, generate multiple candidate mazes,
then select layouts whose topology best matches the target profile.

Primary reference:

> Paul Hyunjin Kim, Jacob Grove, Skylar Wurster, and Roger Crawfis. 2019.
> Design-Centric Maze Generation. In *Proceedings of the 14th International
> Conference on the Foundations of Digital Games* (FDG '19), Article 83,
> 9 pages. https://doi.org/10.1145/3337722.3341854

The paper studies mazes as graph/topology objects and emphasizes controllable
properties such as solution-path length, turns, decision points, dead ends,
runs, and higher-level design intent. This repository does not reimplement the
paper's learned spanning-tree generator. Instead, it uses the same design-centric
workflow in a simpler engineering form suitable for AntMaze:

1. Define target profiles such as `large-like` and `hard`.
2. Randomly generate many connected candidate maps.
3. Score candidates by graph metrics and reject undesirable shapes.
4. Register selected maps as train/eval variants.

## Files

- `generate_antmaze_layouts.py`: candidate generator and profile-based selector.
- `utils/maze_metrics.py`: graph metrics and `static_difficulty` computation.
- `inspect_antmaze_layouts.py`: CLI for registered AntMaze layout metrics.
- `data/antmaze/variants.py`: official and local/test AntMaze variant registry.
- `local_antmaze_gen.py`: offline dataset generator for registered local AntMaze
  variants, using the official Farama AntMaze waypoint-controller stack.
- `generated_antmaze_layouts_seed42.json` and
  `generated_antmaze_layouts_seed42.py`: current generated layout record.

## Layout Profiles

`generate_antmaze_layouts.py` currently defines two profiles.

`large-like` targets maps around the official AntMaze large scale. It allows
moderate loops and a small number of room-like open blocks so the maps do not
become too narrow or brittle.

`hard` targets larger or more constrained layouts with longer paths, more
dead ends, more junctions, and no `2x2` open blocks. The hard profile preserves
long straight corridors when they appear, but rejects room-like open areas such
as `2x2`, `2x3`, and `3x2` contiguous free-space windows.

Current suite layout split:

- `local-layout-01..05`: `large-like`
- `local-layout-06..09`: `hard`
- `test-layout-01..02`: held-out `large-like`
- `test-layout-03..04`: held-out `hard`

## Generation Algorithm

The generator works on a rectangular grid where `1` is wall and `0` is free
space.

1. Sample rows, columns, target free density, and loop probability from the
   profile.
2. Build a full-map room grid using odd interior coordinates.
3. Run randomized depth-first spanning-tree carving over the room grid.
4. Add extra non-tree room edges with the sampled loop probability.
5. Add pocket cells until the target free density is approached.
6. Score the resulting maze and keep the best candidate.

The selector combines several penalties:

- profile mismatch: shortest path, static difficulty, minimum dead ends, minimum
  junctions, and wall density;
- coverage problems: empty interior rows/columns and large solid wall windows;
- open-space problems: `2x2`, `2x3`, and `3x2` contiguous free-space windows.

The open-space penalty is intentionally separate from straight-corridor length.
Long corridors are allowed; room-like open blocks are not desired for the hard
held-out layouts.

Generate the full registered suite:

```bash
python generate_antmaze_layouts.py \
  --mode suite \
  --seed 42 \
  --candidates-per-layout 500 \
  --json-output generated_antmaze_layouts_seed42.json \
  --python-output generated_antmaze_layouts_seed42.py
```

Generate disposable candidates for visual review:

```bash
python generate_antmaze_layouts.py \
  --mode candidates \
  --profile hard \
  --count 4 \
  --candidates-per-layout 800 \
  --seed 2027
```

The generated Python output is a copy/paste aid for `data/antmaze/variants.py`;
the generator does not edit the registry automatically.

## Topology Metrics

`utils/maze_metrics.py` treats each free cell as a graph node and each
four-neighbor free connection as an undirected graph edge. It validates that
the map is rectangular, boundary-walled, and connected.

Core metrics:

- `rows`, `cols`, `free_cells`, `wall_density`
- `shortest_path_len`, `normalized_path_len`, `diameter`
- `solution_turns`, `solution_turn_rate`, `solution_junctions`
- `degree_1_deadends`, `degree_2_corridors`, `degree_3_forks`,
  `degree_4_crosses`, `junction_count`
- `cycle_rank`
- `articulation_points`, `bridge_edges`
- `corridor_count`, `mean_corridor_len`, `max_corridor_len`
- `static_difficulty`

For AntMaze local/test variants, `metrics_for_variant(...)` uses the registered
eval map and the fixed `r/g` cells. For maps without markers, `compute_maze_metrics`
chooses the graph diameter endpoints as start and goal.

Inspect registered AntMaze layouts:

```bash
python inspect_antmaze_layouts.py --variants \
  medium-play large-play \
  local-layout-01 local-layout-02 local-layout-03 \
  test-layout-03 test-layout-04
```

Write JSON:

```bash
python inspect_antmaze_layouts.py --json --output layout_metrics.json
```

## Static Difficulty Formula

`static_difficulty` is a heuristic topology score in `[0, 100]`. It is not an
official D4RL score and is not a rollout success-rate estimate.

The score is:

```text
100 * (
  0.28 * path_component
  + 0.14 * turn_component
  + 0.14 * solution_junction_component
  + 0.12 * articulation_component
  + 0.10 * deadend_component
  + 0.10 * bridge_component
  + 0.07 * wall_component
  + 0.05 * cycle_component
)
```

Components are clamped to `[0, 1]`:

```text
direct_scale = rows + cols - 4
path_component = shortest_path_len / (2.5 * direct_scale)
turn_component = (solution_turns / shortest_path_len) / 0.45
solution_junction_component = (solution_junctions / (shortest_path_len + 1)) / 0.25
articulation_component = (articulation_points / free_cells) / 0.30
deadend_component = (deadends / free_cells) / 0.25
bridge_component = (bridge_edges / edge_count) / 0.60
wall_component = wall_density / 0.65
cycle_component = (cycle_rank / free_cells) / 0.20
```

Interpretation:

- high `path_component`: the start-goal route is long relative to map scale;
- high `turn_component`: the route changes direction often;
- high `solution_junction_component`: the route crosses many decision points;
- high `articulation_component` and `bridge_component`: the graph has many
  bottlenecks;
- high `deadend_component`: more traps or exploratory branches;
- high `wall_component`: denser walls;
- high `cycle_component`: more alternate loops.

Because this is a topology score, it can rank a smaller map above a larger map
when the smaller map has more relative turns or bottlenecks. For example,
official AntMaze/PointMaze `medium` can score slightly above `large` under
this formula if the chosen start-goal route in `large` is relatively direct.

## Current AntMaze Layout Summary

The current registered AntMaze local/test/experimental layouts have these static
scores:

```text
local-layout-01  57.07
local-layout-02  52.62
local-layout-03  48.21
local-layout-04  47.55
local-layout-05  54.13
local-layout-06  66.68
local-layout-07  68.85
local-layout-08  61.49
local-layout-09  73.47

test-layout-01   57.71
test-layout-02   51.52
test-layout-03   70.45
test-layout-04   63.59

ultra            46.06
```

For comparison under the same fixed eval-map scoring:

```text
medium-play      64.91
large-play       63.01
```

## Offline Data Generation

After registering a local AntMaze layout, generate offline data before training:

```bash
python local_antmaze_gen.py \
  --variants local-layout-01 \
  --target-episodes 1000 \
  --num-workers 4 \
  --mode diverse \
  --diverse-cell-mode all-free \
  --min-success-rate 0.8 \
  --seed 42 \
  --overwrite
```

Harder reset/goal sampling is available for local AntMaze diverse data:

```bash
python local_antmaze_gen.py \
  --variants local-layout-01 \
  --target-episodes 1000 \
  --num-workers 4 \
  --mode diverse \
  --diverse-cell-mode all-free \
  --hard-sample \
  --hard-retry 5 \
  --hard-sample-alpha 1.0 \
  --hard-sample-top-n 0 \
  --min-success-rate 1.0 \
  --seed 42 \
  --overwrite
```

The generator loads Farama's official AntMaze `WaypointController` from the
vendored `third_party/minari-dataset-generation-scripts` submodule and uses the
default `GoalReachAnt_model.zip` SAC goal-reaching policy. It writes final data
to:

```text
local_datasets/antmaze-<variant>-v0
```

`--mode diverse` defaults to `--diverse-cell-mode all-free`: the collection map
does not use `c` markers, so Gymnasium Robotics samples reset and goal from all
free cells. `--diverse-cell-mode representative-c` restores the narrower
representative-cell behavior by marking deterministic free cells as `c` and
sampling reset/goal from those combined cells. `--mode play` also leaves every
free cell eligible as a reset/goal candidate. Both modes keep official AntMaze
fixed-horizon episode semantics: reaching the goal records `info["success"]`
but does not truncate the episode.

With `--hard-sample`, the generator precomputes all reachable ordered start/goal
cell pairs from the current diverse candidate set. For `all-free`, this means
all free cells; for `representative-c`, this means the deterministic
representative cells. Each pair receives:

- `path_len`: shortest-path length in grid steps
- `away_steps`: shortest-path steps that increase Manhattan distance from the
  goal
- `away_frac = away_steps / path_len`
- `difficulty = 0.5 * (path_len / max_path_len) + 0.5 * away_frac`

Pairs are then sorted by `difficulty` from low to high and assigned a rank score:

```text
rank_score = rank / max(pair_count - 1, 1)
sample_weight = 1.0 + hard_sample_alpha * rank_score
```

`--hard-sample-top-n 0` keeps every reachable pair. A positive value keeps only
the top N hardest pairs after difficulty sorting and ignores the rest before
rank scores and sampling weights are assigned.

`--hard-sample-alpha 0` is uniform over reachable pairs. `alpha=1` makes the
highest-ranked pair twice as likely as the lowest-ranked pair; `alpha=0.5`
makes it 1.5 times as likely. Larger values bias sampling more strongly toward
longer and more indirect pairs. For each sampled pair, the environment is reset
with fixed `reset_cell` and `goal_cell`. The generator tries the pair at most
`1 + hard_retry` times with different seeds, saves the episode only if it
succeeds, and otherwise discards the failed attempts. It then continues sampling
pairs until exactly `--target-episodes` successful episodes have been saved.

`--min-success-rate` defaults to `0`, which preserves the original behavior of
stopping once `--target-episodes` has been collected. When set above `0`, each
worker first records all episodes until the target count is reached. If the
saved successful-episode ratio is still below the threshold, the worker keeps
sampling: failed post-target episodes are discarded, and each successful
post-target episode replaces one randomly selected failed episode from the
saved set. The final saved dataset therefore keeps exactly `--target-episodes`
episodes unless generation fails. The attempt cap defaults to
`target_episodes * 5` and can be overridden with `--max-episode-attempts`.
Hard-sample mode is different: it only saves successful episodes and ignores
`--max-episode-attempts`, so the practical success-rate setting is
`--min-success-rate 1.0`.

Each generated dataset writes `generation_summary.json` next to the Minari data
directory with the saved dataset success rate (`success_rate` /
`saved_success_rate`) and the empirical success rate over all attempted
episodes before failed-episode filtering (`true_success_rate`). In hard-sample
mode, the summary also records pair-space difficulty stats, saved-episode
difficulty stats, the hard top-N setting, reachable/used pair counts, pair
sampling probability min/max/ratio, hard pair attempt/success/exhaustion
counts, and one `episode_difficulty` entry per saved episode. Use `--overwrite`
when enforcing `--min-success-rate` or regenerating hard-sample data on a
dataset path that already contains episodes.

On Slurm, the repository provides:

```bash
sbatch sbatch/dataGen.ant.slurm
sbatch sbatch/dataGen.ant.hard.slurm
```

`sbatch/dataGen.ant.hard.slurm` covers `local-layout-01..09` as an array job and
defaults to `TARGET_EPISODES=2000`, `MIN_SUCCESS_RATE=1.0`, `HARD_RETRY=5`, and
`HARD_SAMPLE_ALPHA=1.0`. It also accepts `HARD_SAMPLE_TOP_N`, defaulting to `0`
to use all reachable pairs.

The pre-hard-sample initial AntMaze local datasets are backed up under:

```text
local_dataset_backups/antmaze_pre_hard_sample_initial_2026-07-01
```

That backup contains `antmaze-local-layout-01-v0` through
`antmaze-local-layout-09-v0` plus a short `README.txt`. The backup root is
ignored by git.

Training consumes only the generated Minari/HDF5 data. It does not regenerate
trajectories automatically.

## Limitations

- The generator is profile-guided random search, not an optimizer with formal
  guarantees.
- `static_difficulty` is useful for sorting and sanity checks, but it is not a
  substitute for empirical Ant rollout success rate.
- The score depends on the start/goal pair. AntMaze local/test variants use
  fixed registered `r/g` cells; unmarked maps use graph diameter endpoints.
- Official PointMaze normalized score remains separate and is implemented in
  `score.py`.
