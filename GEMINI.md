# Gemini CLI Interaction Notes

This document outlines specific considerations and practices for interacting with the Gemini CLI within this project.

## Worktree Management

Ensure you are operating within the `phase1` worktree for all development and tooling improvements. You can verify your current worktree by running `git worktree list`.

## Tooling Interaction

The `replace` tool is currently non-functional. For all file modifications and interactions, utilize the `desktop_commander` tools (e.g., `desktop_commander__read_file`, `desktop_commander__write_file`, `desktop_commander__edit_block`).
