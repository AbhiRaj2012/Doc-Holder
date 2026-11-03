# SDLC Pipeline: File-by-File Revision and Full Run

Air-Gapped Agentic AI SDLC Automation Pipeline, with the DeepEval evaluation layer applied.

## File-by-file

### config.yaml

- **Takes:** nothing. It is read by `load_config()` and written by `save_config()`.
- **Does:** holds the `ollama` (model, ctx, temperature, server env), `aider`, `workflow` (paths, max iterations, timeouts, truncation limits) and `evaluation` settings.
- **Produces:** the single `cfg` dict that every module reads.

### requirements.txt

- **Takes:** nothing.
- **Does:** lists the dependencies: aider-chat, langgraph, langchain-ollama, deepeval, streamlit and others.
- **Produces:** the package set for the offline cp312 wheel bundle.

### app.py (Streamlit UI)

- **Takes:** sidebar settings, requirements text, user request and the human-gate button clicks.
- **Does:**
  - Sets the offline env vars.
  - Saves the config and starts Ollama.
  - Builds the graph once per session.
  - Creates a fresh `thread_id` per run.
  - Streams node logs through `run_graph`.
  - Renders logs, task stack, diff, reviewer feedback, the eval table and the benchmark HTML.
  - Handles the three gate buttons through `update_state` followed by resume.
- **Produces:** the initial `AgentState`, human decisions injected into the graph, and the on-screen monitoring.

### core/config\_loader.py

- **Takes:** `config.yaml`.
- **Does:** loads and saves the config. Each load or save also rewrites Aider's model-settings YAML so `num_ctx` and `temperature` reach Ollama.
- **Produces:** the `cfg` dict and `workspace_state/.aider.model.settings.yml`.

### core/ollama\_manager.py

- **Takes:** `cfg["ollama"]`.
- **Does:**
  - Exports the `OLLAMA_*` env vars.
  - Pings `/api/tags`, and runs `ollama serve` if the server is down.
  - Checks that the model is installed and auto-pulls it if allowed.
  - `list_local_models` lists the installed models.
- **Produces:** `(ok, message)` and the list of model names.

### core/state.py

- **Takes:** nothing.
- **Does:** defines the pydantic schemas `TaskItem`, `TaskPlan` and `ReviewResult` for structured LLM output. It also defines `AgentState`, which holds the task stack, iteration counters, diff, feedback, logs, human decision, `planned_tasks`, `task_results`, `tool_trace`, `active_sec` and `eval_summary`.
- **Produces:** the types shared by the graph and the nodes.

### core/aider\_bridge.py

- **Takes:** `cfg`, target files, instructions and reviewer feedback.
- **Does:**
  - `_prepare_model` registers the model settings and sets `OLLAMA_API_BASE`.
  - `build_aider_repomap_manifest` walks the repo, ignoring junk files, and builds a Tree-sitter RepoMap.
  - `run_aider_editor` creates an Aider Coder (diff format, no auto-commit, no shell suggestions, no URL fetching), runs the prompt, and keeps chat history in `workspace_state/`.
- **Produces:** `manifest.json` with the file list and repo map, and edits applied directly to the target repo's files.

### core/safety.py

- **Takes:** the project root, LLM-supplied file paths and the verification command.
- **Does:**
  - `safe_target_files` blocks path escape and `.git`.
  - `parse_verification_cmd` allows only whitelisted commands, rejects shell operators and `python -c`, and returns an argv list.
  - `truncate` keeps the head and tail of long text.
- **Produces:** sanitized paths, a safe argv or `None`, and shortened text.

### core/gitutil.py

- **Takes:** the project path.
- **Does:** `ensure_git_repo` makes a baseline commit if there is no repo. `working_diff` runs `add -N` then `diff`, excluding `.aider*`. `stage_all` stages everything, so completed tasks no longer show in later diffs.
- **Produces:** a per-task unified diff string.

### core/agents.py (graph nodes)

- **Takes:** `AgentState` plus a fresh `load_config()` per node.
- **Does / produces, node by node:**
  - `analyzer`: loads or rebuilds the manifest, with a project-root check. Produces `manifest_data`.
  - `planner`: sends the LLM the manifest and request, and gets a `TaskPlan` back. Sanitizes the paths and renumbers the IDs. Produces `task_stack` and `planned_tasks`.
  - `pop_next_task`: takes stack\[0\] and resets the counters. Produces `current_task`.
  - `editor`: runs Aider, records a tool-trace entry (action, tool, status, error) and takes the working diff. Produces `last_editor_diff` and `tool_trace`.
  - `reviewer`: rejects an empty diff, runs the allowlisted command (through the venv python), and asks the LLM for a verdict. Forces fail on a non-zero exit. Produces `task_completed` and `reviewer_feedback`.
  - `complete_task`: pops the task, stages its changes and records a result of `passed` or `skipped`. Rebuilds the manifest. Produces `task_results`.
  - `save_issue`: writes `issue.json`.
  - `evaluator`: marks any unfinished task `failed` and calls `run_evaluation`. Produces `eval_summary`.

### core/graph.py

- **Takes:** the node functions.
- **Does:** wires the StateGraph, with routers after the reviewer, the human gate and the stack check. It wraps nodes in `timed` to accumulate `active_sec`, pauses before `human_gate`, and uses a `MemorySaver` checkpointer.
- **Produces:** the compiled graph.

### core/evaluation.py

- **Takes:** `cfg`, the LLM, `planned_tasks`, `task_results` and `tool_trace`.
- **Does:**
  - Scores six agentic GEval metrics (Plan Quality, Plan Adherence, Tool Correctness, Argument Correctness, Step Efficiency, Task Completion), with deterministic fallbacks.
  - Scores per-file Correctness and Completeness, plus a custom score based on attempts.
  - Classifies each score into Bad, Moderate or Good against its threshold.
  - Appends to the history and renders the HTML and SVG.
- **Produces:**
  - `workspace_state/benchmark_history.jsonl`
  - `workspace_state/benchmark_history.html`
  - `workspace_state/runs/<ts>/evaluation.{json,md,html}`
  - the `eval_summary` dict

## One full run

1. **Start.** You click Start. `ensure_ollama_ready` brings up Ollama. `app.py` builds the initial state with a new `thread_id` and streams the graph.
2. **analyzer.** It ensures the git baseline and loads or builds the RepoMap manifest.
3. **planner.** The LLM turns your request into atomic tasks, for example "TASK-1 edit `validators.py`" and "TASK-2 create `tests/test_validators.py`". The paths are sanitized, and `task_stack` and `planned_tasks` are saved.
4. **pop\_next\_task.** TASK-1 becomes active with the counters reset.
5. **editor.** Aider edits the target files using the instructions plus any reviewer feedback. A trace entry and the diff are recorded.
6. **reviewer.** The pipeline runs the task's verification command, the LLM judges the diff plus the test output, and an exit code other than 0 forces a fail.
   - **Pass:** go to `complete_task`.
   - **Fail with iterations left:** go back to `editor`, with the feedback fed to Aider.
   - **Fail at max iterations:** `save_issue` writes `issue.json`, then the graph pauses at `human_gate`.
7. **Human gate (only on escalation).** The UI shows buttons:
   - **Retry (with hint):** back to `editor`, with the counter reset.
   - **Proceed:** go to `complete_task`, which records the task as `skipped`.
   - **Abort:** go straight to `evaluator`.
8. **complete\_task.** The task is popped, its changes are staged, the manifest is rebuilt and a result is saved.
9. **Loop.** If tasks remain, go back to `pop_next_task` (step 4). Otherwise go to `evaluator`.
10. **evaluator.** It scores the run, appends a record to the JSONL, regenerates the HTML and writes the per-run reports.
11. **Done.** The UI shows the logs, the final diff, the eval table, and the Benchmark History expander with downloads.
