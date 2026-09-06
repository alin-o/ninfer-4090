---
trigger: model_decision
description: Mandatory routing, dependency, deduplication, and specification rules when creating board tasks.
---

# Task Creation Rules

These rules are permanent and must be followed every time you create board tasks.

## Project Routing

- **Determine the correct `project_id` before creating any task.** Group chat
  can coordinate several projects, but each task must target its owning project.
- Each coder agent sandbox has access to exactly ONE repo. Route tasks to the project that owns the code.
- **NEVER mix code from two projects in one task.** A task touching code in two repos MUST be split into two separate tasks, one per project.
- **Before creating tasks, check existing tasks** in the target project to avoid duplicates — especially if a prior run already created tasks for the same work.

## Dependencies First

- Before creating tasks, **analyze the dependency graph**. Don't just convert a user's list into independent parallel tasks.
- Set `depends_on_ids` correctly on creation — blocked tasks should not run until their dependencies complete.
- If unsure about dependencies, err on the side of adding them. A task waiting a few extra minutes is better than one failing because a prerequisite isn't done.

## No Ordinal References

- **Never reference tasks by ordinal** ("task #2", "task #4") in descriptions — the coding agent has zero context for those.
- Always use **actual task IDs** (e.g., `task-1773643043213-vkf51z`) or describe the concrete artifact or interface the dependency produces.
- If a task depends on output from another task, describe the exact file path, function name, or API endpoint — not a position in your mental list.

## Quality Over Speed

- Think before creating tasks. Read the relevant code first.
- Write task descriptions that are self-contained — the coding agent should be able to implement the task with only the description, no external context.
- Include specific file paths, line numbers, function names, and acceptance criteria.

## Tasks in Question

- Treat Question as temporary routing when the blocker is recoverable.
- Resolve reversible internal blockers within the authorized task scope, add
  the answer or evidence to the task, and return it to an executable stage.
- Leave a task in Question only when it genuinely requires external
  information, authority, credentials, or a user decision. State the exact
  requirement and notify the user when appropriate.
