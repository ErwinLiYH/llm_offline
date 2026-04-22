---
name: project-changelog
description: Use this skill when the user asks to update the changelog, record recent project changes, summarize recent modifications before logging them, or check whether AGENTS.md and DESIGN.md should be updated alongside CHANGELOG.md in the llm_offline project. This includes prompts such as "update changelog", "记录改动", "更新changelog", or similar requests to capture recent repo changes into project documentation.
---

# Project Changelog

## Overview

Use this skill when the user wants to record recent project changes into `CHANGELOG.md`.

In this repo, changelog updates should be grounded in actual recent work, not generic summaries. The workflow is:
- inspect current context and recent repo changes
- compare against existing `CHANGELOG.md` to avoid duplicates
- give the user a concise change summary first
- append new changelog content to the bottom of `CHANGELOG.md`
- scan `AGENTS.md` and `DESIGN.md` to see whether the recent changes should also be reflected there

## Trigger Phrases

This skill should be used for requests like:
- update changelog
- 更新 changelog
- 更新CHANGELOG
- 记录这次改动
- 记录最近的修改
- 把这些变化写进 changelog

## Workflow

1. Inspect the current work before editing documentation.
Check at minimum:
- `git status --short`
- relevant changed files
- the tail of `CHANGELOG.md`
- `AGENTS.md`
- `DESIGN.md`

2. Compare recent work against `CHANGELOG.md`.
Do not duplicate items that are already recorded. Prefer using the current conversation context plus actual changed files to infer what is new.

3. Before writing, give the user a concise summary of the changes you believe should be logged.
Keep that summary short and high-signal.

4. Append the new changelog entry to the bottom of `CHANGELOG.md`.
Do not insert the entry in the middle of the file. Preserve the existing writing style and chronological ordering already used in the file.

5. After drafting the changelog entry, scan `AGENTS.md` and `DESIGN.md`.
Only update them if the recent changes materially affect:
- architecture or data flow
- config semantics
- prompt system behavior
- evaluation or training workflow
- project-specific operating guidance

6. If `AGENTS.md` or `DESIGN.md` do not need updates, say so briefly instead of forcing a documentation edit.

## Changelog Rules

- Append at the end of `CHANGELOG.md`, never in the middle.
- Group related changes into a few sections instead of one bullet per file.
- Focus on behavior changes, interfaces, configs, prompt semantics, evaluation behavior, and debugging tools.
- Do not log ephemeral artifacts such as local result folders, temporary experiment outputs, or cache files unless the user explicitly asks.
- If a change is still exploratory and not actually implemented in the repo, do not log it as completed work.
- If there is no substantive new change compared with the existing changelog, say that explicitly and do not add a duplicate entry.

## Repo-Specific Notes

- `CHANGELOG.md` is the canonical change log file.
- `AGENTS.md` is the main project-operating guide for Codex in this repo.
- `DESIGN.md` is the design reference and may lag behind implementation; only update it when implementation meaningfully changes documented behavior.
- Recent PointMaze changes often touch `train.py`, `evaluate.py`, `data/pointmaze/`, `prompts/pointmaze/`, and config files. Check those first when inferring what to log.
