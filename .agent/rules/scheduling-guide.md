---
trigger: model_decision
description: Guidance for choosing scheduled tasks, cronjobs, and dataset tasks.
---

# Scheduling: When to Use What

Choose by the artifact the user expects:

| Mechanism | Creates | Scope | Use for |
|---|---|---|---|
| Scheduled chat task | An agent run in chat | Group | Reminders, summaries, and conversational checks |
| Cronjob | One board card per occurrence | Project | Repeatable work that needs visible task history |
| Dataset task | One board card per data row | Project | Batch work where each item needs separate tracking |

Scheduled chat prompts must be self-contained because the scheduled run may not
receive the current conversation. If it must contact the user, explicitly tell
it to call `send_message` with the correct group folder.

Use the scheduling tool's schema for supported schedule formats and timezone
semantics. Do not invent fields or calculate from stale assumptions.

CarapaBrain objective triggers are a separate objective-only mechanism. A group
agent may explain the distinction but does not configure them as group chat
scheduling.
