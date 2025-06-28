# Gemini CLI Interaction Notes

This document outlines specific considerations and practices for interacting with the Gemini CLI within this project.

## Workflow
If you are changing code as result of a task, always run the associated tests. If there are no tests, ask the user whether to create tests for the changed code or not. 

## Worktree Management

Ensure you are operating within the correct worktree for all development and tooling improvements. You can verify your current worktree by running `git worktree list`.

## Tooling Interaction

The `replace` tool is currently non-functional. For all file modifications and interactions, utilize the `desktop_commander` tools (e.g., `desktop_commander__read_file`, `desktop_commander__write_file`, `desktop_commander__edit_block`).

## Collaboration
After you are finished, push to a branch associated with your task. Make sure not branch exists for this task. Create a PR using the github cli. Check the status of the github cicd using the cli, and fix any issues present. If there are any ci/cd problems, evaluate whether they can be checked locally, such as with the pre-commit or by building/installing the package. 