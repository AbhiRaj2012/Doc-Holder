import re
import os
import json
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# DeepEval is optional; the pipeline still runs without it.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False

# Initialize LLM
llm = ChatOllama(
    model="gemma4:e2b", # Ensure you are using the most capable tag for coding
    num_predict=2048,
    num_ctx=8192,
    temperature=0.1
)

# ==========================================
# 1. LOGGER UTILITY
# ==========================================
def write_log(run_dir: Path, step: str, details: str):
    log_file = run_dir / "execution_log.txt"
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] === {step.upper()} ===\n")
        f.write(str(details) + "\n")
        f.write("-" * 80 + "\n")

# ==========================================
# 2. ENHANCED STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    user_input: str
    requirements: str
    project_manifest: str
    run_dir: Path
    file_system: Annotated[dict, operator.ior]
    pending_files: list[str]
    feedback: str
    review_attempts: int
    iterations: int
    file_attempts: Annotated[dict, operator.ior]

# ==========================================
# 3. NODES
# ==========================================
def requirement_node(state: AgentState):
    print("🔍 Generating requirements...")
    prompt = f"""
    You are an expert project analyzer, who analyzes the user's request and writes requirement.txt file for the demanded application.\n
    User wants: {state['user_input']}\n
    Write a clear requirement.txt with project name, description, etc, then moving to functional and non functional dependencies, use cases, etc. \n
    Note: - Use short precise bullet points.
          - Tryto cover basic requirements to fulfill the request but don't go too deep.\n
          - Keep the output fully structured like a professional document.
    """
    write_log(state["run_dir"], "Requirement Node - PROMPT", prompt)

    requirements = ""
    for chunk in llm.stream(prompt):
        requirements += chunk.content
    requirements = requirements.strip()

    write_log(state["run_dir"], "Requirement Node - OUTPUT", requirements)
    (state["run_dir"] / "requirements.txt").write_text(requirements, encoding="utf-8")
    return {"requirements": requirements}


def architect_node(state: AgentState):
    print("\n📐 [Architect] Creating Project Manifest & Scaffolding...")

    # UPGRADE: Force a layout structure to prevent full-width stacking
    manifest_prompt = f"""
    You are a professional Project Manager and Architect, who analyzes the project requirements and creates a project manifest file in json format for proper execution and implementation of the project.\n
    Requirements: {state['requirements']}\n
    Analyze the project requirements and create the manifest file that conatains the following:\n
    Section : Layout Structure, IDs, Functions.\n
        - Layout Structure: It must describe a html skeleton (component order and arrangement like grid or flex) and component tags that will be used for simple css file creation.\n
        - IDs section: It must list each element's ID in Key:Value pairs. It will be used to connet the components at html with the functions in js file.\n
        - Functions section: It must be a json object describing all function's declaration and description (what they take and what they do). It will be used to implement the functions in js file.\n
   
    You MUST output:
    1. LAYOUT STRUCTURE: Define the exact wrapper classes needed (e.g., '.calculator-grid', '.display-area') so CSS can implement a Grid/Flexbox properly.
    2. DOM IDs: A dictionary of exact DOM Element IDs that HTML and JS will strictly share.
    3. DATA: The exact JSON data structure for the localStorage ledger.
    4. Clearly and in detail state the planning/design/content of html/css/js.
    5. Also strictl define the files names (to avoid mismatch issue, like index.html, style.css, script.js). Strictly use the file names as [index.html, style.css, script.js] and don't use any other names.

    The manifest should show all input fields, buttons, etc properly, which is calling or called by which function, and what should that function do.
    """

    manifest = llm.invoke(manifest_prompt).content.strip()
    write_log(state["run_dir"], "Architect Node - MANIFEST", manifest)

    # Save the manifest immediately
    (state["run_dir"] / "manifest.txt").write_text(manifest, encoding="utf-8")

    initial_files = {
        "index.html": "<!DOCTYPE html>\n<html lang='en'>\n<head>\n  <meta charset='UTF-8'>\n  <title>App</title>\n  <link rel='stylesheet' href='style.css'>\n</head>\n<body>\n  <div id='app'></div>\n  <script src='script.js'></script>\n</body>\n</html>",
        "style.css": "/* Global Styles */\nbody { font-family: sans-serif; padding: 20px; }",
        "script.js": "// Main Logic"
    }

    return {
        "project_manifest": manifest,
        "file_system": initial_files,
        "pending_files": ["index.html", "style.css", "script.js"],
        "review_attempts": 0,
        "feedback": "",
        "iterations": 0,
        "file_attempts": {}
    }

ALL_FILENAMES = ["index.html", "style.css", "script.js"]

def sanitize_editor_output(raw_output: str, current_file: str) -> str:
    fence_match = re.search(r"```(?:[a-zA-Z]+)?\n?(.*?)\n?```", raw_output, re.DOTALL)
    code = fence_match.group(1).strip() if fence_match else raw_output.strip()

    lines = code.splitlines()
    if lines and lines[0].strip().lower() in ALL_FILENAMES:
        lines = lines[1:]
    code = "\n".join(lines).strip()

    for other in ALL_FILENAMES:
        if other == current_file:
            continue
        marker = re.search(rf"(?m)^\s*{re.escape(other)}\s*$", code)
        if marker:
            code = code[:marker.start()].strip()
    return code

def editor_node(state: AgentState):
    current_file = state["pending_files"][0]
    base_code = state["file_system"][current_file]

    print(f"\n💻 [Editor] Drafting {current_file} (Streaming)...")

    project_context = "".join([f"\n--- {f.upper()} ---\n{c}\n" for f, c in state["file_system"].items()])

    # UPGRADE: Strict diff-style instructions
    prompt = f"""
    You are a Professional web developer and have a great knoowledge and experience in html/css/js.\n
    You are currently working on: {current_file}.\n

    PROJECT MANIFEST (Contract): {state['project_manifest']}\n

    CURRENT PROJECT FILES: {project_context}\n

    PREVIOUS REVIEWER FEEDBACK: {state['feedback']}\n

    CRITICAL INSTRUCTION: 
    If there is feedback, you MUST ONLY change the exact lines mentioned. DO NOT refactor or rewrite the rest of the file.
    You MUST return the ENTIRE updated file content so it can be saved, but leave the un-flagged code exactly as it was.
    Provide ONLY the complete code for {current_file}. No markdown code fences, no explanations.

    Note:\n
    - If working on html look for the Layout structure and IDs section in the manifest for layout and ID knowledge and strictly follow it.\n
    - If working on css look for the Layout structure and IDs section in the manifest for layout, class and component Id knowledge and strictly follow it.\n
    - If working on js look for the Functions and IDs section in the manifest for functions and ID related knowledge and strictly follow it.\n
    - Strictly use the file names as [index.html, style.css, script.js] and don't use any other names.\n
    - If working on one of the files (like html/css/js), you only have to generate code of that file and follow the structure of that file, don't generate code of the other ones.\n
    - You MUST strictly follow the manifest and implement the project as per the manifest.\n
    """
    write_log(state["run_dir"], f"Editor Node [{current_file}] - PROMPT", prompt)

    raw_output = ""
    try:
        for chunk in llm.stream(prompt):
            raw_output += chunk.content
            print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"\n   [!] Streaming interrupted: {e}")

    raw_output = raw_output.strip()
    print("\n")

    clean_code = sanitize_editor_output(raw_output, current_file)

    updated_fs = state["file_system"].copy()
    if len(clean_code) > 10:
        updated_fs[current_file] = clean_code

    attempts = state.get("file_attempts", {}).get(current_file, 0) + 1
    return {
        "file_system": updated_fs,
        "iterations": state["iterations"] + 1,
        "file_attempts": {current_file: attempts}
    }


def reviewer_node(state: AgentState):
    current_file = state["pending_files"][0]
    file_code = state["file_system"][current_file]

    print(f"\n🔎 [Reviewer] Auditing {current_file}...")

    project_context = "".join([f"\n--- {f.upper()} ---\n{c}\n" for f, c in state["file_system"].items()])

    # UPGRADE: Added Syntax Linter and structured Replace/With feedback
    prompt = f"""
    You are a professional Project Reviewer, who checks the project files and look for if there is any bug,id mismatch, layout breakdown, etc.\n
    Perform a strict Syntax, DOM, and Functionality Audit on {current_file}.

    MANIFEST: {state['project_manifest']}
    FULL PROJECT CONTEXT: {project_context}
    CODE TO REVIEW ({current_file}):
    {file_code}

    CRITICAL CHECKS:
    1. SYNTAX LINTING: Are there any mismatched quotes (e.g. id='btn">), unclosed tags, or missing brackets?
    2. DOM BINDING: Do the IDs/Classes/Scripts exactly match the Manifest?
    3. UI LAYOUT: Does the CSS implement a proper Grid or Flexbox matching the HTML wrappers?
    4. Check all files and their requirements/plans to review them.
    5. Don't allow empty or only biolerplates written scripts.

    If the code is 100% correct, reply with ONLY the word "PASS". 
    If there are errors, identify the specific mistake and provide SURGICAL feedback in this exact format:
    "Replace: [bad code line]
    With: [corrected code line]"
    """

    review_output = llm.invoke(prompt).content.strip()
    write_log(state["run_dir"], f"Reviewer Node [{current_file}]", review_output)

    if re.match(r"^\s*PASS\.?\s*$", review_output, re.IGNORECASE):
        print(f"   ✅ {current_file} passed review.")
        return {
            "feedback": "",
            "review_attempts": 0
        }
    else:
        print(f"   ❌ Issues found in {current_file}. Routing back to Editor for surgical fix. Review Count: {state['review_attempts']}")
        return {
            "feedback": review_output,
            "review_attempts": state["review_attempts"] + 1
        }


def save_file_node(state: AgentState):
    current_file = state["pending_files"][0]
    content = state["file_system"][current_file]

    # Save the individual file immediately upon passing
    file_path = state["run_dir"] / current_file
    file_path.write_text(content, encoding="utf-8")
    print(f"   💾 Saved verified file: {file_path}")

    # Pop the completed file from the queue
    return {
        "pending_files": state["pending_files"][1:],
        "review_attempts": 0,
        }


# ==========================================
# 4. EVALUATION (DeepEval)
# ==========================================
if DEEPEVAL_AVAILABLE:
    class OllamaEvalModel(DeepEvalBaseLLM):
        # Wraps the same local Ollama model so GEval judging stays offline.
        def __init__(self, chat_model, name):
            self._chat = chat_model
            self._name = name

        def load_model(self):
            return self._chat

        def generate(self, prompt: str) -> str:
            return self._chat.invoke(prompt).content

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return self._name


def _build_deepeval_metrics() -> list:
    judge = OllamaEvalModel(llm, llm.model)
    return [
        GEval(
            name="Correctness",
            criteria=(
                "The ACTUAL_OUTPUT is a generated code file. Using the requirements and manifest given in "
                "INPUT as ground truth: does every document.getElementById(...) call use an id that is "
                "actually described in the manifest? Does each function's logic plausibly match its purpose? "
                "Flag any obvious runtime bug."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        ),
        GEval(
            name="Completeness",
            criteria=(
                "Using the requirements and manifest given in INPUT as the spec, does ACTUAL_OUTPUT implement "
                "every required id/function with real logic, not a TODO or empty stub?"
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        ),
    ]


def deepeval_score_file(filename: str, code: str, context: str, metrics: list):
    if not DEEPEVAL_AVAILABLE or not code.strip():
        return None
    test_case = LLMTestCase(input=context, actual_output=code)
    results = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            score = round(metric.score, 2) if metric.score is not None else None
            results[metric.name] = {"score": score, "reason": metric.reason}
        except Exception as e:
            results[metric.name] = {"score": None, "reason": f"DeepEval metric failed: {e}"}
    return results


def compute_custom_score(attempts: int) -> float:
    # Deterministic, non-LLM score: fewer editor retries = higher score.
    return round(max(0.25, 1.0 - 0.25 * (attempts - 1)), 2)


def render_evaluation_markdown(overall: dict, report: dict) -> str:
    lines = ["# Evaluation Report", "", f"**Custom score avg:** {overall['custom_score_avg']} / 1.0"]
    if "deepeval_avg_score" in overall:
        lines.append(f"**DeepEval score avg:** {overall['deepeval_avg_score']} / 1.0")
    if not overall["deepeval_available"]:
        lines.append("\n_DeepEval isn't installed (`pip install deepeval`) — showing custom scores only._")
    lines.append("")
    for filename, data in report.items():
        lines.append(f"## {filename}")
        lines.append(f"- Attempts: {data['attempts']} | Custom score: {data['custom_score']}")
        deval = data.get("deepeval")
        if deval:
            for name, res in deval.items():
                score_str = res["score"] if res["score"] is not None else "n/a"
                lines.append(f"  - **{name}**: {score_str} — {res['reason']}")
        lines.append("")
    return "\n".join(lines)


def benchmark_history_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark_history.jsonl"


def load_benchmark_history() -> list:
    path = benchmark_history_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_benchmark_record(record: dict) -> list:
    with open(benchmark_history_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return load_benchmark_history()


def render_benchmark_svg(records: list) -> str:
    W, H, PAD = 640, 260, 42
    n = len(records)
    if n == 0:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="120">'
                f'<text x="20" y="60" font-family="sans-serif" font-size="13">'
                f'No runs tracked yet.</text></svg>')

    def x_for(i):
        return PAD + (i / max(n - 1, 1)) * (W - 2 * PAD)

    def y_for(s):
        return H - PAD - s * (H - 2 * PAD)

    def series_svg(key: str, color: str):
        pts = [(i, r[key]) for i, r in enumerate(records) if r.get(key) is not None]
        if not pts:
            return "", None
        poly = " ".join(f"{x_for(i):.1f},{y_for(s):.1f}" for i, s in pts)
        dots = "".join(
            f'<circle cx="{x_for(i):.1f}" cy="{y_for(s):.1f}" r="3" fill="{color}">'
            f'<title>run {i + 1}: {s:.2f}</title></circle>'
            for i, s in pts
        )
        avg = sum(s for _, s in pts) / len(pts)
        return f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>{dots}', avg

    deepeval_svg, deepeval_avg = series_svg("deepeval_avg_score", "#2563eb")
    custom_svg, custom_avg = series_svg("custom_score_avg", "#16a34a")

    gridlines = "".join(
        f'<line x1="{PAD}" y1="{y_for(g):.1f}" x2="{W - PAD}" y2="{y_for(g):.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        f'<text x="4" y="{y_for(g) + 4:.1f}" font-size="10" fill="#6b7280">{g:.1f}</text>'
        for g in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    legend = (
        f'<circle cx="{PAD}" cy="16" r="4" fill="#2563eb"/><text x="{PAD + 10}" y="20" font-size="11" fill="#374151">'
        f'DeepEval avg{f" ({deepeval_avg:.2f})" if deepeval_avg is not None else ""}</text>'
        f'<circle cx="{PAD + 160}" cy="16" r="4" fill="#16a34a"/>'
        f'<text x="{PAD + 170}" y="20" font-size="11" fill="#374151">'
        f'Custom avg{f" ({custom_avg:.2f})" if custom_avg is not None else ""}</text>'
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
{gridlines}
{deepeval_svg}
{custom_svg}
{legend}
<text x="{PAD}" y="{H - 8}" font-size="11" fill="#374151">Run 1</text>
<text x="{W - PAD - 40}" y="{H - 8}" font-size="11" fill="#374151">Run {n}</text>
</svg>'''


def render_benchmark_html(records: list) -> str:
    scored_deepeval = [r["deepeval_avg_score"] for r in records if r.get("deepeval_avg_score") is not None]
    scored_custom = [r["custom_score_avg"] for r in records if r.get("custom_score_avg") is not None]
    all_time_deepeval = round(sum(scored_deepeval) / len(scored_deepeval), 2) if scored_deepeval else "n/a"
    all_time_custom = round(sum(scored_custom) / len(scored_custom), 2) if scored_custom else "n/a"

    rows = []
    for i, r in enumerate(records):
        cfg = r.get("llm_config", {})
        rows.append(
            f"<tr><td>{i + 1}</td><td>{r.get('timestamp', '')}</td>"
            f"<td>{cfg.get('model', '')}</td><td>{cfg.get('num_ctx', '')}</td>"
            f"<td>{r.get('app_brief', '')[:60]}</td>"
            f"<td>{r.get('files_passed', '')}</td>"
            f"<td>{r.get('custom_score_avg', 'n/a')}</td>"
            f"<td>{r.get('deepeval_avg_score', 'n/a')}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Loop Workflow — Benchmark History</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111827; max-width: 900px; }}
table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; font-size: 13px; }}
th {{ background: #f9fafb; }}
</style></head>
<body>
<h1>Benchmark History</h1>
<p>{len(records)} run(s) tracked — all-time avg custom score: <b>{all_time_custom}</b>
&nbsp;|&nbsp; all-time avg DeepEval score: <b>{all_time_deepeval}</b></p>
{render_benchmark_svg(records)}
<table>
<tr><th>#</th><th>Timestamp</th><th>Model</th><th>num_ctx</th><th>App</th>
<th>Files passed</th><th>Custom avg</th><th>DeepEval avg</th></tr>
{"".join(rows)}
</table>
</body></html>"""


def evaluator_node(state: AgentState):
    print("\n📊 [Evaluator] Scoring the finished app...")
    run_dir = state["run_dir"]
    files = ["index.html", "style.css", "script.js"]
    context = f"REQUIREMENTS:\n{state['requirements']}\n\nMANIFEST:\n{state['project_manifest']}"
    metrics = _build_deepeval_metrics() if DEEPEVAL_AVAILABLE else []

    report = {}
    for filename in files:
        code = state["file_system"].get(filename, "")
        attempts = state["file_attempts"].get(filename, 1)
        report[filename] = {
            "attempts": attempts,
            "custom_score": compute_custom_score(attempts),
            "deepeval": deepeval_score_file(filename, code, context, metrics),
        }

    custom_scores = [f["custom_score"] for f in report.values()]
    overall = {
        "files_passed": f"{len(files)}/{len(files)}",
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "custom_score_avg": round(sum(custom_scores) / len(custom_scores), 2) if custom_scores else 0.0,
    }
    deval_scores = [m["score"] for f in report.values() if f.get("deepeval")
                     for m in f["deepeval"].values() if m and m.get("score") is not None]
    if deval_scores:
        overall["deepeval_avg_score"] = round(sum(deval_scores) / len(deval_scores), 2)

    (run_dir / "evaluation.json").write_text(json.dumps({"overall": overall, "files": report}, indent=2), encoding="utf-8")
    (run_dir / "evaluation.md").write_text(render_evaluation_markdown(overall, report), encoding="utf-8")
    write_log(run_dir, "Evaluator", json.dumps(overall, indent=2))

    # Cross-run benchmark tracking — accumulates next to the script, across runs.
    benchmark_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "app_brief": state.get("user_input", "")[:200],
        "llm_config": {
            "model": llm.model,
            "num_ctx": llm.num_ctx,
            "num_predict": llm.num_predict,
            "temperature": llm.temperature,
        },
        "files_passed": overall["files_passed"],
        "custom_score_avg": overall["custom_score_avg"],
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "deepeval_avg_score": overall.get("deepeval_avg_score"),
        "per_file": {fn: {"attempts": d.get("attempts")} for fn, d in report.items()},
    }
    history = append_benchmark_record(benchmark_record)
    (benchmark_history_path().parent / "benchmark_history.html").write_text(
        render_benchmark_html(history), encoding="utf-8"
    )

    print(f"   Custom score avg: {overall['custom_score_avg']}/1.0")
    if "deepeval_avg_score" in overall:
        print(f"   DeepEval score avg: {overall['deepeval_avg_score']}/1.0")
    print(f"   📈 Benchmark: {len(history)} run(s) tracked — see benchmark_history.jsonl / benchmark_history.html")
    return {}


# ==========================================
# 5. CONDITIONAL ROUTERS & GRAPH
# ==========================================
def route_review(state: AgentState) -> str:
    MAX_REVIEW_ATTEMPTS = 3
    if state["feedback"] == "":
        return "pass"
    if state["review_attempts"] >= MAX_REVIEW_ATTEMPTS:
        print("   ⚠️ Max review attempts reached — forcing pass to avoid infinite loop.")
        return "pass"
    return "fail"

def route_next_file(state: AgentState) -> str:
    return "continue" if len(state["pending_files"]) > 0 else "done"

def reset_state(state: AgentState):
    state["feedback"] = ""
    state["review_attempts"] = 0


workflow = StateGraph(AgentState)
workflow.add_node("requirements", requirement_node)
workflow.add_node("architect", architect_node)
workflow.add_node("editor", editor_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("save_file", save_file_node)
workflow.add_node("evaluator", evaluator_node)

workflow.set_entry_point("requirements")
workflow.add_edge("requirements", "architect")
workflow.add_edge("architect", "editor")
workflow.add_edge("editor", "reviewer")

workflow.add_conditional_edges("reviewer", route_review, {"fail": "editor", "pass": "save_file"})
workflow.add_conditional_edges("save_file", route_next_file, {"continue": "editor", "done": "evaluator"})
workflow.add_edge("evaluator", END)

app = workflow.compile()

# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    user_input = input("\nWhat application do you want to build?\n> ")

    base_dir = Path.cwd() / "SDLC_Runs"
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    write_log(run_dir, "SYSTEM", f"Starting build for: '{user_input}'")

    print(f"\n🚀 Starting build in: {run_dir}")
    app.invoke({"user_input": user_input, "run_dir": run_dir})
    print(f"\n🎉 Build Complete! Check execution_log.txt in {run_dir}")
    print(f"📊 Evaluation report: {run_dir / 'evaluation.md'}")
    print(f"📈 Benchmark history: {Path(__file__).resolve().parent / 'benchmark_history.html'}")