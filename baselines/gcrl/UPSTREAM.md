# Upstream provenance

The state-based CRL and HIQL losses and the small Flax network/train-state
subset in this directory are adapted from OGBench's `impls` reference code:

- Repository: <https://github.com/seohongpark/ogbench>
- Commit: `1d4140997f60c52c6fb0702ec100dc988b18c548`
- OGBench version at that commit: `1.2.1`
- Source files: `impls/agents/{crl,hiql}.py` and
  `impls/utils/{datasets,encoders,flax_utils,networks}.py`
- License: MIT, copyright 2024 OGBench Authors

Local changes split CrossMaze states from compact goals, preserve goal
relabeling within each maze variant, add deterministic rollout prediction,
and integrate the agents with this repository's data/evaluation artifacts.
