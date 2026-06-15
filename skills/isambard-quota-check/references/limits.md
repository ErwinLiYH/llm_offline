# Isambard Limits Reference

Use live cluster commands for current values. This file records stable
interpretation rules and documentation links, not current usage.

## Storage

| Area | Purpose | Important behavior |
|---|---|---|
| `$HOME` | Personal persistent files, source, environments | Capacity and file-count quotas; not a backup |
| `$SCRATCHDIR` | High-performance temporary project work | Lustre project quota; check bytes and inodes |
| `$PROJECTDIR` | Shared persistent project data | Shared project quota, not personal capacity |
| `$LOCALDIR` | Node-local temporary I/O | Cleared after the job/session; never use as the only copy |

The authoritative Lustre check is:

```bash
project_id=$(lfs project -d "$SCRATCHDIR" | awk 'NR == 1 {print $1}')
lfs quota -hp "$project_id" "$SCRATCHDIR"
```

One file, directory, or symbolic link normally consumes one inode. Many small
files can exhaust inode quota while byte use remains low.

## Slurm And Credits

- Query live QoS limits with `sacctmgr show qos` and account associations with
  `sacctmgr show assoc`.
- Isambard accounts compute usage in node-hour-rate credits (NHR).
- Requested resources and requested walltime reserve credits. Oversized
  walltime requests can reduce the ability to start other jobs.
- Current credit balance is normally checked in the BriCS Portal, not inferred
  from `squeue` or `sacct`.
- Interactive jobs may have separate QoS limits and accounting.
- Use checkpoints for jobs approaching the live QoS walltime limit.

## Operational Rules

- Keep heavy computation and deep filesystem scans off login nodes.
- Avoid frequent scheduler polling; use reasonable intervals for scripted
  `squeue` checks.
- Home, scratch, and project storage are not archival backups.
- Retrieve important data before project expiry. Retention after project end is
  policy-controlled and should be confirmed in the current FAQ.
- For static datasets containing very many small files, consider SquashFS or
  another packed representation.

## Official Documentation

- Storage: https://docs.isambard.ac.uk/user-documentation/information/system-storage/
- Job scheduling: https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/
- How Isambard works and NHR: https://docs.isambard.ac.uk/user-documentation/information/how-isambard-works/
- Slurm guide: https://docs.isambard.ac.uk/user-documentation/guides/slurm/
- SquashFS datasets: https://docs.isambard.ac.uk/user-documentation/tutorials/datasets-in-squashfs/
- FAQ and retention: https://docs.isambard.ac.uk/user-documentation/faqs/

Browse these official pages when the user requests current policy details or
when a documented value may have changed.
