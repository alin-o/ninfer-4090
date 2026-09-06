---
trigger: model_decision
description: CarapaBoard MCP tools and required project routing for group operations.
---

## MCP Tools Available

You connect to CarapaBoard through tools prefixed
`mcp__carapa-board__`. Use this group's folder for chat, memory, and scheduled
prompts. Use the owning project's ID for project-scoped board operations.

| Tool | Purpose |
|------|---------|
| `send_message` | Send a message to this group's chat |
| `create_task` | Create a kanban task card |
| `list_tasks` | List kanban tasks (filter by project, stage) |
| `get_task` | Get full task details by ID |
| `update_task` | Update task status, title, description, priority |
| `add_task_comment` | Leave a note on a task |
| Scheduling tools | Manage scheduled chat prompts, cronjobs, and dataset tasks |
| Memory tools | Save, search, and remove durable group memories |

Tool availability and schemas are authoritative. Do not infer objective-only
operations from tools that happen to be available.
