# Repository Agent Instructions

## Mandatory plan maintenance

The single master plan is `docs/MASTER_PLAN.md`. These instructions apply to every agent working in this repository, including delegated agents and reviewers.

Before starting work, read the master plan and the relevant linked protocol. Identify the parent plan, task ID, current status, owner, and evidence. Before executing, claim an existing task row or add a scoped child task with an owner and acceptance condition. Do not restart completed work or invent a new stage numbering scheme.

Maintain the master plan throughout the task. Update it when work starts, scope changes, an experiment fails, computation finishes, validation passes, or artifacts are backed up. Distinguish execution status from scientific conclusions. Never mark running work complete or infer hypothesis success from a completed run.

Place higher-level plan tables before lower-level plans. Add new plans under their proper parent. Preserve completed, failed, paused, and superseded rows with completion date, result, limitations, and links to evidence. Do not delete completion records or silently rewrite old protocols. Record plan changes in the change log.

Every agent is responsible for maintaining its own task rows. Before each edit, re-read the latest master plan, then apply a targeted change; do not overwrite another agent's updates with an old full-file copy. If direct writing is unavailable, send the parent agent a precise proposed update and have it merged and verified. Read-only auditors may report their update for the parent to merge.

Before ending, update task and parent status, record remaining work honestly, and verify evidence links. The root agent reconciles parallel updates. Do not create competing master plans elsewhere. `Agent.md` is a pointer to these canonical instructions, not a second rule source.

## Research and execution constraints

Use Chinese for user-facing reports. Preserve negative results. Keep exploratory analysis, independent confirmation, weakly supervised intervention fitting, and unsupervised discovery separate. Previously inspected test sets are not fresh confirmation data. Do not claim physical dynamics from static images or require JEPA to outperform other encoders.

Use only the authorized single H100 for current remote experiments; select an available device and record it rather than hardcoding a CUDA assumption in reusable code. Keep core code and necessary results in `F:\2610\CVPR27`, with source/configuration/data hashes and validation evidence. Never store passwords or tokens in repository files or logs.

Do not push to GitHub without a new explicit user instruction to push. Preserve existing files and historical experiment records.
