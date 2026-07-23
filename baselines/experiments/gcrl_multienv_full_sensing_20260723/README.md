# Full-sensing multi-environment GCRL experiment

This is a controlled follow-up to
`gcrl_multienv_paper_20260722`.  CRL and HIQL retain the same OGBench-derived
algorithm parameters, 3x512 GELU+LayerNorm networks, seed, 1M-update budget,
variant sampler, train/test split, and checkpoint cadence.  The only model
input change is that all three numeric CrossMaze features are enabled:

- `include_map`: a row-major 0/1 map with `-1` padding to the largest map in
  the same environment family.  It is appended to the state, so held-out map
  layouts are observable at rollout time without changing state dimensionality.
- `include_location_sensing`: current grid cell is appended to state and goal
  grid cell to the separately conditioned goal vector.  Goal relabeling
  recomputes the goal cell from the relabelled xy target.
- `include_wall_sensing`: four state-local `[up, down, left, right]` values
  (`0=free`, `1=wall`, `2=risk`) under the repository's `v3` sensing contract.

`balance_variant_episode_count` remains `false` deliberately: this round
isolates observability from the separately proposed sampler-rebalancing
intervention.  Raw binary artifacts are under `baseline_runs/` and are not
tracked.  The audit and rendered report are generated after all four formal
runs finish.

```bash
# PointMaze CRL (replace crl with hiql as needed).
micromamba run -n llm_offline_gcrl python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml baselines/configs/crl.yaml \
  baselines/experiments/gcrl_multienv_paper_20260722/pointmaze.yaml \
  baselines/experiments/gcrl_multienv_paper_20260722/paper_protocol.yaml \
  baselines/experiments/gcrl_multienv_full_sensing_20260723/full_sensing.yaml \
  --experiment_id gcrl-full-sensing-crl-pointmaze-multienv-s0-1m-20260723

# Verify every checkpoint and write the Markdown report after all four runs.
micromamba run -n llm_offline_gcrl python \
  baselines/experiments/gcrl_multienv_full_sensing_20260723/audit_results.py
```
