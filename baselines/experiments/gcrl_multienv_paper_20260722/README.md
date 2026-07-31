# Formal multi-environment GCRL experiment

> Historical protocol: these runs used the retired `compact_xy` goal semantics.
> Their recorded numbers remain unchanged but are not directly comparable to
> `full_observation_v1` runs or loadable by the current GCRL checkpoint path.

This protocol trains each GCRL algorithm separately on PointMaze and AntMaze
multi-environment suites.  Held-out `test-layout*` maps are never supplied to
the offline sampler.  They appear only in checkpoint rollout evaluation.

The full algorithm/family matrix is:

| Family | Train variants | Held-out rollout variants | Runs |
| --- | --- | --- | --- |
| PointMaze | `open`, `umaze`, `medium`, `large`, `local-layoutV2-01..12` | `test-layoutV2-01..06` | CRL, HIQL |
| AntMaze | `umaze`, `medium-diverse`, `large-diverse`, `ultra`, `local-layout-01..12` | `test-layout-01..04` | CRL, HIQL |

At each saved checkpoint (250k, 500k, 750k, 1m updates), every train and test
variant is rolled out for 30 fixed reset seeds.  This creates 660 PointMaze and
600 AntMaze episodes per checkpoint, per algorithm.  The audit script verifies
the 30-episode contract and reports train/test aggregates separately.

```bash
# Example: PointMaze CRL formal run.
micromamba run -n llm_offline_gcrl python baseline_train.py --config \
  baselines/configs/base.pointmaze.yaml baselines/configs/crl.yaml \
  baselines/experiments/gcrl_multienv_paper_20260722/pointmaze.yaml \
  baselines/experiments/gcrl_multienv_paper_20260722/paper_protocol.yaml \
  --experiment_id gcrl-paper-crl-pointmaze-multienv-s0-1m-20260722

# After all four runs finish.
micromamba run -n llm_offline_gcrl python \
  baselines/experiments/gcrl_multienv_paper_20260722/audit_results.py
```
