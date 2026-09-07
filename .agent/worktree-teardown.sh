#!/bin/sh
set -eu

# No services or external per-worktree resources are created by setup.
# Carapa owns checkout removal/reset; preserve the shared compiler cache.
exit 0
