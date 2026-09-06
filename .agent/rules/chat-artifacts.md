---
trigger: model_decision
description: How to return generated files and screenshots through chat.
---

# Chat Artifacts

- Do not present an arbitrary absolute filesystem path as though it were a
  user-downloadable chat link.
- To expose a generated file, place it in the configured group artifact
  directory, normally `<GROUP_DIR>/artifacts/`, and return the resulting
  artifact reference.
- Put screenshots and other images under the artifact directory's `images/`
  subdirectory. Inline image rendering does not by itself create a durable
  downloadable artifact.
