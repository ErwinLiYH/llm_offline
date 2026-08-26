# CrossMaze Topology Difficulty v3.3

`crossmaze.topology_difficulty` implements the formal whole-map topology
difficulty metric. It is independent from both:

- `crossmaze.eval_position` path/map difficulty v2, used for eval hard-pair
  selection and hardest-10%-path map metadata;
- legacy `utils.maze_metrics.static_difficulty`, which remains unchanged for
  existing map-generation and inspection workflows.

## Public API

```python
from crossmaze import compute_topology_difficulty

result = compute_topology_difficulty(maze_map)
print(result.version)                 # v3.3
print(result.raw_score)               # weighted score before Logistic
print(result.score)                   # final 0-100 score
print(result.metrics.top_level_path_count)
print(result.to_dict())
```

The complete fixed scoring configuration is available through:

```python
from crossmaze import topology_difficulty_config
```

## Top-level paths

The maze is an undirected four-neighbor graph over free cells. For every
unordered free-cell pair, the implementation retains one canonical shortest
path:

1. cells and pair endpoints are sorted;
2. BFS visits neighbors in `up, down, left, right` order;
3. a path and its reversal are treated as the same path.

A path is contained by another path only when its complete cell sequence is a
contiguous subpath of the other path, in either direction. A top-level path is
a maximal path under this containment relation. This is sequence containment,
not set containment.

## Raw metrics

For the top-level-path set `P`, v3.3 computes:

- `N`: top-level path count;
- `L`: mean path length in grid edges;
- `R`: mean direction-change count;
- `D`: mean detour ratio, where each path uses
  `1 - Manhattan(endpoint_pair) / path_length`;
- `A`: full matrix area `rows * cols`, including boundary walls;
- `T`: degree-3 free cells whose four diagonals are all walls;
- `X`: degree-4 free cells whose four diagonals are all walls.

Requiring all diagonals to be walls prevents open-room degree-3/4 cells from
being counted as corridor T/cross junctions.

## Fixed normalization

```text
c_N = min(1, log(1 + N) / log(101))
c_L = L / (L + 10)
c_R = R / (R + 3)
c_D = D * c_L
c_A = A / (A + 100)
c_T = T / (T + 5)
c_X = X / (X + 2)
```

`c_D` is detour exposure rather than bare detour ratio: a high proportional
detour on a very short path is deliberately discounted.

Every normalized component then receives the same fixed contrast transform:

```text
G(c) = c^1.5 / (c^1.5 + (1-c)^1.5)
```

The transform preserves 0, 0.5, and 1, lowers values below 0.5, and raises
values above 0.5. It is fixed and does not depend on which map set is scored.

## Raw and final score

The v3.3 weights are:

| Component | Weight |
|---|---:|
| Top-level path count | 5% |
| Mean path length | 25% |
| Mean turns | 15% |
| Mean detour exposure | 25% |
| Map area | 20% |
| Corridor T junctions | 5% |
| Corridor cross junctions | 5% |

```text
raw_score = 100 * (
    0.05 * G(c_N)
  + 0.25 * G(c_L)
  + 0.15 * G(c_R)
  + 0.25 * G(c_D)
  + 0.20 * G(c_A)
  + 0.05 * G(c_T)
  + 0.05 * G(c_X)
)

score = 100 / (1 + exp(-(raw_score - 45) / 10))
```

The final Logistic transform is monotonic, so it expands score separation
without changing raw-score ordering. The unrounded raw score is passed to the
Logistic function; returned `raw_score` and `score` are each rounded to two
decimal places.

## Complexity

Let `V` be the free-cell count, `E` the graph edge count,
`P=V(V-1)/2` the unordered pair count, and `D_max` the graph diameter.

- all-pairs canonical shortest paths:
  `O(V * (V + E) + P * D_max)`;
- containment marking: approximately `O(P * D_max^2)` candidate slices plus
  tuple hashing/copying;
- stored pair paths: `O(P * D_max)` cells in the coarse worst case.

The implementation avoids a direct `O(P^2)` comparison between all path pairs.

## Limitations

- Only one BFS-tie-broken shortest path is retained per unordered pair.
- Changing neighbor order can change containment, turns, and the final score.
- Top-level path count reflects canonical-path selection as well as graph
  topology; this is why its weight remains small.
- Junction counts are global unique-cell counts, not averages along paths.
- All constants are fixed design parameters rather than fitted statistics.

