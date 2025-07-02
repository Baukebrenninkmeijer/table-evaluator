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


## Gemini CLI Integration

### Overview
You can offload individual tasks to the Gemini CLI, which you can call with the `gemini` command.
Always use YOLO mode (`-y`) for trusted development environments to streamline the workflow.

### Command Reference
```bash
gemini -p "your prompt here" -y
```

### Proven Use Cases & Best Practices

#### ✅ **Highly Effective Tasks**
- **Project Status Analysis**: Comprehensive project reports and progress summaries
- **CI/CD Pipeline Optimization**: Performance analysis and configuration improvements
- **Code Analysis & Optimization**: Benchmark suite enhancements and refactoring suggestions
- **Documentation Generation**: Technical documentation and README sections

#### ⚠️ **Tasks with Limitations**
- **Large Codebase Analysis**: Tasks with `--all_files` may timeout after 2 minutes
- **Complex Multi-step Operations**: Break into smaller, focused tasks
- **Import Path Dependencies**: May require manual corrections for project-specific imports

### Recommended Workflow

1. **Start with Focused Tasks** (< 500 words prompts)
   ```bash
   gemini -p "Analyze CI pipeline and suggest 3 specific optimizations" -y
   ```

2. **Use YOLO Mode for Trusted Environments**
   ```bash
   gemini -p "Optimize benchmark performance tests" -y
   ```

3. **Validate and Test Gemini Suggestions**
   - Always run local tests after Gemini modifications
   - Check import paths and dependencies
   - Verify configuration changes

4. **Combine with Manual Implementation**
   - Use Gemini for analysis and recommendations
   - Implement complex changes manually with Gemini insights
   - Leverage Gemini's optimization suggestions

### Success Metrics from Polars Integration Project

- **CI/CD Pipeline**: 40-50% performance improvement through parallel execution and caching
- **Benchmark Suite**: 60% execution time reduction via module-scope fixtures
- **Code Organization**: Enhanced test structure with pytest markers and categories
- **Documentation Quality**: Comprehensive technical documentation and status reports

### Options Reference
```
Options:
  -m, --model                    Model      [string] [default: "gemini-2.5-pro"]
  -p, --prompt                   Prompt. Appended to input on stdin (if any).
                                                                        [string]
  -s, --sandbox                  Run in sandbox?                       [boolean]
      --sandbox-image            Sandbox image URI.                     [string]
  -d, --debug                    Run in debug mode?   [boolean] [default: false]
  -a, --all_files                Include ALL files in context?
                                                      [boolean] [default: false]
      --show_memory_usage        Show memory usage in status bar
                                                      [boolean] [default: false]
  -y, --yolo                     Automatically accept all actions (aka YOLO
                                 mode, see
                                 https://www.youtube.com/watch?v=xvFZjo5PgG0 for
                                 more details)?       [boolean] [default: false]
      --telemetry                Enable telemetry? This flag specifically
                                 controls if telemetry is sent. Other
                                 --telemetry-* flags set specific values but do
                                 not enable telemetry on their own.    [boolean]
      --telemetry-target         Set the telemetry target (local or gcp).
                                 Overrides settings files.
                                              [string] [choices: "local", "gcp"]
      --telemetry-otlp-endpoint  Set the OTLP endpoint for telemetry. Overrides
                                 environment variables and settings files.
                                                                        [string]
      --telemetry-log-prompts    Enable or disable logging of user prompts for
                                 telemetry. Overrides settings files.  [boolean]
  -c, --checkpointing            Enables checkpointing of file edits
                                                      [boolean] [default: false]
  -v, --version                  Show version number                   [boolean]
  -h, --help                     Show help                             [boolean]
  ```
