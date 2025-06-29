# AI Assistant Development Workflow Guidelines

This document defines the standard workflow for AI-assisted development tasks within this project.

## Development Workflow

### Code Changes and Testing
- **Always run tests** after implementing code changes
- If no tests exist for modified code, ask whether to create them before proceeding
- Use `make test` or equivalent project-specific test commands

### Branch and Worktree Management
- Verify you're in the correct worktree: `git worktree list`.
- Verify you're in the correct branch: `git branch`.
- If no worktree and branch exists for this task, create them.
- Operate within the appropriate branch worktree for all development tasks
- Ensure branch isolation for different features/tasks

### Collaboration and Integration
1. **Branch Creation**: Check if task branch exists; create if needed
2. **Code Push**: Push changes to feature branch upon completion
3. **Pull Request**: Create PR using GitHub CLI (`gh pr create`)
4. **CI/CD Monitoring**:
   - Monitor pipeline status: `gh pr checks`
   - Address failures promptly
   - Run local equivalents when possible (pre-commit, build, install)
5. **Quality Gates**: Ensure all checks pass before requesting review

## Local Development Best Practices
- Run pre-commit hooks before committing
- Verify builds/installations locally to catch CI/CD issues early
- Use project's Makefile targets for standardized operations
