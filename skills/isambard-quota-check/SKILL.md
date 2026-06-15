---
name: isambard-quota-check
description: Audit Isambard storage and compute limits using live cluster commands and official documentation. Use when the user asks to check, summarize, diagnose, or monitor quota, disk usage, inode usage, home/scratch/project/local storage, Slurm job or QoS limits, walltime, NHR credits, project expiry, retention, or errors such as "Disk quota exceeded" on Isambard.
---

# Isambard Quota Check

## Workflow

1. Run the bundled read-only audit:

```bash
bash skills/isambard-quota-check/scripts/audit_isambard_quota.sh
```

If invoked outside this repository, resolve the script relative to this `SKILL.md`.

2. Read [references/limits.md](references/limits.md) when interpreting policy,
retention, NHR credits, or documentation defaults.

3. Report these separately:

- live storage use and hard limits for home, scratch project, project storage,
  and local node storage
- block capacity and inode/file-count capacity
- largest top-level directories
- live Slurm account, QoS, walltime, running-job, and submitted-job limits
- policy-only limits that cannot be queried locally, especially NHR credit
  balance and project expiry

4. Prefer live cluster output over documented defaults when they differ.
State both values and the date of the check. Do not infer that a zero quota
field means unlimited when the path inherits a project or default quota.

5. Diagnose `Disk quota exceeded` by checking both bytes and inodes. On Lustre,
query the path's project ID with `lfs project -d`, then query that project with
`lfs quota -p`; a user quota alone may hide the active hard limit.

6. Keep the final answer compact but include:

- a table of current use versus hard limits
- percentage or remaining headroom where available
- warnings ordered by practical risk
- exact commands for future checks
- official documentation links for policy claims

## Safety

- Treat the audit as read-only.
- Do not delete caches, logs, datasets, or checkpoints without explicit user
  authorization.
- Do not run expensive recursive scans on login nodes beyond the bounded
  top-level checks in the bundled script unless the user asks for deeper
  analysis.
- Do not claim to know the current NHR credit balance from Slurm. Direct the
  user to the BriCS Portal unless another authoritative live source is
  available.
