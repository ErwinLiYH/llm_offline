# Official Maze Dataset Semantics

This document records the practical differences between official Minari/D4RL
PointMaze and AntMaze data, the AntMaze `play`/`diverse` split, and this
repository's current local AntMaze data-generation behavior.

## Sources

The repository vendors Farama's dataset-generation scripts in:

- `third_party/minari-dataset-generation-scripts/scripts/pointmaze/create_pointmaze_dataset.py`
- `third_party/minari-dataset-generation-scripts/scripts/D4RL/antmaze/create_antmaze_dataset.py`
- `third_party/minari-dataset-generation-scripts/scripts/D4RL/antmaze/controller.py`

The local AntMaze generator is:

- `local_antmaze_gen.py`

The AntMaze and PointMaze variant registries are:

- `data/antmaze/variants.py`
- `data/pointmaze/variants.py`

## PointMaze vs AntMaze

Official PointMaze and AntMaze have different episode semantics.

PointMaze is generated as one long continuing trajectory with random goals.
The collection env uses `continuing_task=True`, `reset_target=True`, and a very
large `max_episode_steps`. The dataset callback truncates an episode when the
current navigation goal is reached, so most PointMaze episodes are goal-arrival
segments. The public sparse datasets are effectively successful segments, plus a
small number of residual zero-reward truncated fragments at storage boundaries.

AntMaze is generated with `continuing_task=True` and `reset_target=False`.
Episodes are fixed-horizon environment rollouts. Reaching the goal records
`info["success"]`, but does not end the episode and does not reset the target.
If the ant falls or cannot move, the episode still continues until the TimeLimit
truncates it. Therefore official AntMaze includes failed fixed-horizon episodes.

High-level comparison:

| Property | Official PointMaze | Official AntMaze |
|---|---|---|
| Agent | Point mass | MuJoCo Ant |
| Action dim | 2 | 8 |
| Collection task | Continuing random-goal navigation | Fixed-goal rollout per episode |
| Goal reset after success | Yes | No |
| Episode split | At goal success | At fixed horizon / TimeLimit |
| Failure trajectories | Almost none in sparse public data | Present |
| Fall/unhealthy termination | Not applicable | Not propagated as episode termination |
| Official public steps | 1M per dataset | 1M per dataset |

## Observed Public Dataset Statistics

The following statistics were computed from local Minari downloads under the
`d4rl_datagen` environment. `success_any` for AntMaze means that at least one
step in the episode has `info["success"] == True`. PointMaze success-ending
counts use positive reward on the final transition, because Minari PointMaze
episode `infos` include one extra reset entry.

### AntMaze

| Dataset | Env | Horizon | Episodes | Lengths | Success any | Failed | Last truncated |
|---|---|---:|---:|---|---:|---:|---:|
| `D4RL/antmaze/umaze-v1` | `AntMaze_UMaze-v4` | 700 | 1430 | `500 x 5`, `700 x 1425` | 1311 / 1430 | 119 | 1430 |
| `D4RL/antmaze/umaze-diverse-v1` | `AntMaze_UMaze-v4` | 700 | 1430 | `500 x 5`, `700 x 1425` | 1342 / 1430 | 88 | 1430 |
| `D4RL/antmaze/medium-play-v1` | `AntMaze_Medium-v4` | 1000 | 1000 | `1000 x 1000` | 871 / 1000 | 129 | 1000 |
| `D4RL/antmaze/medium-diverse-v1` | `AntMaze_Medium_Diverse_GR-v4` | 1000 | 1000 | `1000 x 1000` | 827 / 1000 | 173 | 1000 |
| `D4RL/antmaze/large-play-v1` | `AntMaze_Large-v4` | 1000 | 1000 | `1000 x 1000` | 846 / 1000 | 154 | 1000 |
| `D4RL/antmaze/large-diverse-v1` | `AntMaze_Large_Diverse_GR-v4` | 1000 | 1000 | `1000 x 1000` | 840 / 1000 | 160 | 1000 |

All listed AntMaze episodes have `terminated=False` on the final step. The
UMaze datasets contain five shorter `500`-step fragments due to dataset storage
or checkpoint boundaries; the rest are fixed horizon.

### PointMaze

| Dataset | Env | Collection max step | Episodes | Length range | Mean length | Success-ending segments | Residual zero-reward segments |
|---|---|---:|---:|---:|---:|---:|---:|
| `D4RL/pointmaze/open-v2` | `PointMaze_Open-v3` | 1,000,000 | 9525 | 1-300 | 104.99 | 9520 | 5 |
| `D4RL/pointmaze/umaze-v2` | `PointMaze_UMaze-v3` | 1,000,000 | 13210 | 1-190 | 75.70 | 13205 | 5 |
| `D4RL/pointmaze/medium-v2` | `PointMaze_Medium-v3` | 1,000,000 | 4752 | 1-504 | 210.44 | 4747 | 5 |
| `D4RL/pointmaze/large-v2` | `PointMaze_Large-v3` | 1,000,000 | 3360 | 1-797 | 297.62 | 3355 | 5 |

The PointMaze score/evaluation horizons are separate from the collection
`max_episode_steps`: open/umaze use 300, medium uses 600, and large uses 800 in
the official-style score envs.

## AntMaze Play vs Diverse

Both official AntMaze `play` and `diverse` datasets use fixed-horizon episodes
and can contain failures. The main difference is the collection map's reset/goal
candidate set.

For `medium` and `large`:

- `play` uses the base maze map without `r`, `g`, or `c` markers. Gymnasium
  Robotics treats every empty cell as a valid reset and goal candidate.
- `diverse` uses the `_DIVERSE_GR` map with hand-picked `c` cells. Each `c`
  cell is both a reset candidate and a goal candidate.

For `umaze`:

- `umaze` uses a manually supplied fixed reset/goal collection map in the
  official AntMaze generation script.
- `umaze-diverse` uses `AntMaze_UMaze-v4` with no `c` cells, so reset and goal
  are sampled from all free cells.

The `c` cells are defined by Gymnasium Robotics as combined reset/goal cells.
If a map has no explicit `r`, `g`, or `c`, all empty cells become both reset and
goal candidates.

## Current Local AntMaze Generation

`local_antmaze_gen.py` uses the official AntMaze waypoint-controller stack:

- official `WaypointController`
- official `GoalReachAnt_model.zip` SAC goal-reaching policy
- default `--action-noise 0.2`
- Gymnasium Robotics v4 AntMaze observation/action contract

The local generator now follows official AntMaze episode semantics: reaching the
goal does not truncate the episode. An episode ends only through environment
termination/truncation, normally the TimeLimit. This is intentionally different
from PointMaze local generation, where first success is the default truncation
point.

Current local AntMaze modes:

| Local option | Reset/goal candidate source | Uses `c` markers by default | Episode semantics |
|---|---|---:|---|
| `--mode diverse --diverse-cell-mode all-free` | all free cells | No | fixed horizon |
| `--mode diverse --diverse-cell-mode representative-c` | deterministic representative free cells | Yes | fixed horizon |
| `--mode play` | all free cells | No | fixed horizon |
| `--mode diverse --hard-sample` | difficulty-weighted explicit start/goal pairs from the diverse candidate set | follows `--diverse-cell-mode` | fixed horizon; save successes only |

The default is:

```bash
--mode diverse --diverse-cell-mode all-free
```

This differs from official `medium-diverse` and `large-diverse`, which use
hand-picked `c` cells. It is closer to official `play` and `umaze-diverse` in
that reset and goal are sampled from all free cells. The mode name is kept as
`diverse` for experiment organization and because reset/goal pairs are still
diverse; it does not mean the local collection map has official-style `c` cells
unless `--diverse-cell-mode representative-c` is explicitly selected.

The representative `c` mode is not an official hand-picked cell list. It is a
deterministic heuristic over the local maze graph, intended only as a narrower
sampling option when all-free reset/goal sampling is too broad.

Hard-sample mode is also repository-specific. It precomputes all reachable
ordered start/goal cell pairs from the diverse candidate set, scores each pair
with `difficulty`, sorts pairs from low to high difficulty, and then applies a
rank-linear sampling weight:

```text
difficulty = 0.5 * (path_len / max_path_len) + 0.5 * away_frac
rank_score = rank / max(pair_count - 1, 1)
sample_weight = 1.0 + hard_sample_alpha * rank_score
```

where `away_frac` is the fraction of shortest-path steps that move farther away
from the goal in Manhattan distance. `hard_sample_alpha=0` is uniform,
`alpha=1` makes the highest-ranked pair twice as likely as the lowest-ranked
pair, and smaller values such as `0.5` give a gentler bias. The generator then
samples pairs by `sample_weight`, resets the environment with fixed
`reset_cell` and `goal_cell`, and saves only successful episodes. Each pair is
tried at most `1 + hard_retry` times. This is not part of official public
AntMaze dataset generation.

`--hard-sample-top-n N` can restrict sampling to only the top N hardest pairs
after difficulty sorting. `N=0` keeps all reachable pairs.

## Success-Rate Filtering

Official public AntMaze datasets keep the natural success/failure mix produced
by the generator. This repository can optionally enforce a saved success rate:

```bash
--min-success-rate 0.8
```

With this option enabled, the generator first saves the target number of
episodes. If the saved success rate is below the threshold, it keeps attempting
episodes. Post-target failed episodes are discarded; post-target successful
episodes replace randomly selected saved failed episodes. The final dataset
keeps exactly `--target-episodes` saved episodes unless generation fails.

This success-rate filtering is a repository-specific behavior and is not part
of the official public AntMaze dataset generation.

Hard-sample mode bypasses the replacement-style success-rate filtering above:
failed attempts are never saved, successful episodes are appended, and
generation continues until the saved dataset reaches `--target-episodes`.
`--max-episode-attempts` is intentionally ignored in hard-sample mode. In
practice the saved success rate should be `1.0`, so hard-sample Slurm jobs use
`--min-success-rate 1.0`.

Each local AntMaze dataset writes `generation_summary.json`, including:

- `mode`
- `diverse_cell_mode`
- `collection_combined_cells`
- `success_rate` / `saved_success_rate`
- `true_success_rate`
- attempted, discarded, and replacement episode counts

Hard-sample summaries additionally include:

- `hard_sample`, `hard_retry`, `hard_sample_alpha`
- `hard_sample_top_n`, `hard_pair_space_total`, `hard_pair_space_used`
- pair-space `difficulty`, `path_len`, and `away_steps` min/max/mean
- pair sampling probability min/max and max-over-min ratio
- saved-episode `difficulty`, `path_len`, and `away_steps` min/max/mean
- `hard_pairs_sampled`, `hard_pairs_succeeded`, `hard_pairs_exhausted`
- `hard_failed_attempts`
- `episode_difficulty` records with each saved episode's start/goal cells and
  pair difficulty metrics

## Practical Training Implications

For BC training:

- PointMaze official data mostly teaches goal-reaching behavior up to first
  success. It has very little failed behavior.
- AntMaze official data includes both successful and failed fixed-horizon
  behavior. A success can appear before the end of the sequence, followed by
  more transitions around the same fixed target.
- Local AntMaze all-free diverse data has higher reset/goal randomness than
  official medium/large diverse `c`-cell data.
- Enabling `--min-success-rate` makes the saved local AntMaze dataset less
  like the raw official public distribution, but can be useful when a custom
  map/controller combination produces too many failures.
- Enabling `--hard-sample` changes the saved distribution more strongly: it
  over-samples longer or more indirect reachable start/goal pairs and removes
  failed episodes from the saved dataset.
