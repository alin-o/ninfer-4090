---
trigger: always_on
description: Universal container, communication, verification, retry, and user-question behavior.
---

# Tool Behavior Rules

## Container Constraints

- You run inside a Docker container. You cannot execute docker commands, restart containers, or access other services' filesystems.
- Do not attempt `docker`, `curl localhost`, or any host-level operations. They will never work.
- If something needs a container restart or host-level action, tell the user — don't try it yourself.

## Communication Style

- Be short and factual. No filler, no fluff, no "I'll stay here" or "I'll keep that in mind."
- Do not make promises about future behavior — you have no continuity between sessions.
- If you can't guarantee something, say so. Honesty over politeness.
- Never say "it should work" or "the sequence should be" without having verified the actual code path. If you haven't traced the code, say "I haven't verified this."
- When a mechanism fails, do not retry the same approach with a small tweak. Stop and find a fundamentally different solution.

## AskUserQuestion

- After calling `AskUserQuestion`, **stop immediately**. Do not continue with another turn or add follow-up text.
- Wait for the user's response before proceeding.
- The UI renders the question options to the user — no need to repeat or summarize them after the tool call.
