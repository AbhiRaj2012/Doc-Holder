"""
Language-agnostic, LangGraph-driven codebase editor running on local Ollama.

Graph:
    requirement -> analyzer -> planner -> editor -> reviewer --(issue)--> editor
                                            ^                    |
                                            +----(stack left)----+
                                                                 |
                                                       (stack empty) -> evaluator -> END

The planner outputs the plan as a LIFO stack: the first step to execute sits on TOP
(end of the list), so `plan_stack.pop()` always yields the next step in order.
"""
import re
import os
import json
import time
import shutil
import difflib
import argparse
import subprocess
import operator
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, Optional, Tuple, List

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False

# ==========================================
# 0. CONFIG & LLM CLIENTS
# ==========================================
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "30192"))

MAX_REVIEW_ATTEMPTS = 3
MAX_TASKS = 15
MAX_SCAN_FILES = 5000
MAX_TREE_ENTRIES = 1500
MAX_FILE_CHARS = 9000
PLANNER_CONTENT_BUDGET = 9000
MAX_DIFF_CHARS = 6000

# Metric Specifications from Reference Matrix
AGENTIC_METRIC_SPECS = {
    "Plan Quality": {
        "key": "plan_quality",
        "bad_max": 0.59,
        "good_min": 0.80,
        "threshold": 0.70,
        "bad_label": "0 - 0.59",
        "good_label": "0.80 - 1.0",
        "criteria": (
            "Evaluate the quality, granularity, logical ordering, and completeness of the execution plan "
            "in ACTUAL_OUTPUT against the user request and repository manifest in INPUT. "
            "Check if tasks are properly scoped (one file/logical unit per step), ordered so dependencies "
            "are created before use, and cover the full user requirement without redundant steps."
        ),
    },
    "Plan Adherence": {
        "key": "plan_adherence",
        "bad_max": 0.59,
        "good_min": 0.80,
        "threshold": 0.70,
        "bad_label": "0 - 0.59",
        "good_label": "0.80 - 1.0",
        "criteria": (
            "Evaluate how strictly the executed steps, file modifications, and diffs in ACTUAL_OUTPUT "
            "adhered to the planned steps and descriptions specified in INPUT. Penalize skipped tasks, "
            "unauthorized out-of-scope file modifications, or deviations from planned target functions."
        ),
    },
    "Tool Correctness": {
        "key": "tool_correctness",
        "bad_max": 0.69,
        "good_min": 0.90,
        "threshold": 0.70,
        "bad_label": "0 - 0.69",
        "good_label": "0.90 - 1.0",
        "criteria": (
            "Evaluate whether the correct editing action/tool ('edit' via SEARCH/REPLACE blocks vs 'create' "
            "for full file generation) and static verification tools were selected and applied in ACTUAL_OUTPUT "
            "given the file existence state and task requirements in INPUT."
        ),
    },
    "Argument Correctness": {
        "key": "argument_correctness",
        "bad_max": 0.69,
        "good_min": 0.90,
        "threshold": 0.70,
        "bad_label": "0 - 0.69",
        "good_label": "0.90 - 1.0",
        "criteria": (
            "Evaluate the accuracy of tool arguments in ACTUAL_OUTPUT against INPUT: valid relative file paths, "
            "accurate target function/symbol identifiers, verbatim matching SEARCH blocks with proper context lines, "
            "and syntactically valid replacement code blocks."
        ),
    },
    "Step Efficiency": {
        "key": "step_efficiency",
        "bad_max": 0.69,
        "good_min": 0.85,
        "threshold": 0.70,
        "bad_label": "0 - 0.69",
        "good_label": "0.85 - 1.0",
        "criteria": (
            "Evaluate the execution efficiency in ACTUAL_OUTPUT based on total iterations, retry attempts, "
            "and failed edits relative to total tasks in INPUT. First-try passes with minimal diff overhead "
            "and zero redundant retries should score highest (0.85 - 1.0)."
        ),
    },
    "Task Completion": {
        "key": "task_completion",
        "bad_max": 0.69,
        "good_min": 0.90,
        "threshold": 0.70,
        "bad_label": "0 - 0.69",
        "good_label": "0.90 - 1.0",
        "criteria": (
            "Evaluate whether the final code changes and task outcomes in ACTUAL_OUTPUT completely, "
            "accurately, and functionally fulfill the original user request in INPUT, with all tasks passing "
            "review and no remaining syntax errors, TODOs, or broken references."
        ),
    },
}

llm = ChatOllama(model=MODEL_NAME, num_predict=2048, num_ctx=NUM_CTX, temperature=0.1)
llm_json = ChatOllama(model=MODEL_NAME, num_predict=2048, num_ctx=NUM_CTX, temperature=0.1, format="json")
llm_editor = ChatOllama(model=MODEL_NAME, num_predict=4096, num_ctx=NUM_CTX, temperature=0.1)

IGNORE_DIRS = {
    "node_modules", "__pycache__", "venv", "env", "dist", "build", "target", "bin", "obj",
    "vendor", "coverage", "SDLC_Runs", "site-packages",
}

EXT_LANG = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".swift": "Swift", ".go": "Go", ".rs": "Rust", ".c": "C", ".h": "C", ".cpp": "C++", ".cc": "C++",
    ".hpp": "C++", ".cs": "C#", ".php": "PHP", ".rb": "Ruby", ".scala": "Scala", ".dart": "Dart",
    ".lua": "Lua", ".r": "R", ".sh": "Shell", ".bash": "Shell", ".ps1": "PowerShell", ".sql": "SQL",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "CSS", ".sass": "CSS", ".vue": "Vue",
    ".svelte": "Svelte", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
    ".xml": "XML", ".md": "Markdown",
}
SPECIAL_FILENAMES = {"Dockerfile": "Dockerfile", "Makefile": "Makefile"}
DATA_LANGS = {"JSON", "YAML", "TOML", "XML", "Markdown", "Dockerfile", "Makefile"}

FRAMEWORK_SIGNATURES = {
    "package.json": {
        "react": "React", "vue": "Vue", "@angular/core": "Angular", "next": "Next.js", "svelte": "Svelte",
        "express": "Express", "@nestjs/core": "NestJS", "electron": "Electron", "jest": "Jest",
        "vite": "Vite", "typescript": "TypeScript", "tailwindcss": "Tailwind CSS", "react-native": "React Native",
    },
    "requirements.txt": {
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "langgraph": "LangGraph",
        "langchain": "LangChain", "pytest": "pytest", "sqlalchemy": "SQLAlchemy", "pandas": "pandas",
        "numpy": "NumPy", "torch": "PyTorch", "tensorflow": "TensorFlow",
    },
    "pom.xml": {"spring-boot": "Spring Boot", "junit": "JUnit"},
    "build.gradle": {"spring-boot": "Spring Boot", "com.android": "Android", "junit": "JUnit"},
    "build.gradle.kts": {"spring-boot": "Spring Boot", "com.android": "Android", "junit": "JUnit"},
    "Cargo.toml": {"actix-web": "Actix Web", "tokio": "Tokio", "rocket": "Rocket", "axum": "Axum"},
    "go.mod": {"gin-gonic/gin": "Gin", "labstack/echo": "Echo", "gofiber/fiber": "Fiber"},
    "Gemfile": {"rails": "Ruby on Rails", "sinatra": "Sinatra", "rspec": "RSpec"},
    "composer.json": {"laravel/framework": "Laravel", "symfony": "Symfony"},
    "pubspec.yaml": {"flutter": "Flutter"},
}
FRAMEWORK_SIGNATURES["pyproject.toml"] = FRAMEWORK_SIGNATURES["requirements.txt"]
FRAMEWORK_SIGNATURES["Pipfile"] = FRAMEWORK_SIGNATURES["requirements.txt"]

ENTRY_POINT_NAMES = {
    "main.py", "app.py", "manage.py", "wsgi.py", "asgi.py", "__main__.py", "index.js", "server.js", "app.js",
    "index.ts", "main.ts", "main.go", "main.rs", "lib.rs", "Main.java", "Application.java", "Program.cs",
    "index.html", "main.c", "main.cpp", "main.dart", "index.php", "main.rb", "config.ru",
}

DEFAULT_LANGUAGE_GUIDE = (
    "Match the file's existing style exactly: indentation, quoting, naming, import ordering and comment style."
)
LANGUAGE_GUIDES = {
    "Python": "Preserve indentation exactly (spaces vs tabs). Keep imports at the top; add new imports only if required. Follow existing type-hint and docstring conventions.",
    "JavaScript": "Keep the existing module system (ESM vs CommonJS), semicolon and quote style. Never introduce undefined identifiers; keep DOM ids and exports consistent.",
    "TypeScript": "Keep types strict and explicit; do not use `any` unless the file already does. Keep the existing module and semicolon style.",
    "Java": "Keep braces, access modifiers, package and import layout. Update imports when adding types; keep checked exceptions handled.",
    "Kotlin": "Follow existing null-safety and scope-function idioms; keep package and imports consistent.",
    "C#": "Keep namespace, using directives, access modifiers and async/await conventions consistent.",
    "Go": "Output gofmt-style code (tabs). Handle every returned error the way surrounding code does; keep imports minimal and used.",
    "Rust": "Keep ownership and lifetimes correct; propagate errors with the file's existing pattern (Result/?). Keep `use` statements tidy.",
    "C": "Keep header/source declarations in sync; manage memory explicitly and match existing brace style.",
    "C++": "Keep header/source declarations in sync; follow existing RAII and namespace conventions.",
    "PHP": "Keep opening tags, namespaces and PSR style consistent; escape output the way surrounding code does.",
    "Ruby": "Follow existing block and method style; keep `end` balance correct.",
    "HTML": "Keep tags balanced and ids/classes consistent with the linked CSS and JS.",
    "CSS": "Reuse existing selectors and variables; keep specificity and layout (flex/grid) consistent.",
    "Shell": "Keep quoting safe (\"$var\"), preserve the shebang and `set` options.",
    "SQL": "Keep dialect-specific syntax consistent with existing statements; never drop data unless asked.",
}

EDIT_FORMAT = """OUTPUT FORMAT (strict). Return ONLY one or more blocks exactly like this, no commentary, no markdown fences:
<<<<<<< SEARCH
<lines copied VERBATIM from the current file, including indentation>
=======
<the replacement lines>
>>>>>>> REPLACE

Rules:
- SEARCH must match the file character for character and match exactly one location; include 2-3 unchanged neighbouring lines for uniqueness.
- Keep every block minimal. NEVER output the whole file. NEVER change code unrelated to the task.
- To insert code, SEARCH an anchor line and repeat it in REPLACE together with the new lines.
- To delete code, leave the REPLACE section empty."""

CREATE_FORMAT = """OUTPUT FORMAT (strict). Return the COMPLETE new file inside ONE fenced code block and nothing else.
The file must be fully implemented: no TODOs, no placeholders, no empty stubs."""


# ==========================================
# 1. LOGGER & GENERIC UTILITIES
# ==========================================
def write_log(run_dir: Path, step: str, details: str):
    log_file = run_dir / "execution_log.txt"
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] === {step.upper()} ===\n")
        f.write(str(details) + "\n")
        f.write("-" * 80 + "\n")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_llm(client, system: str, human: str, stream: bool = False) -> str:
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    last_err = None
    for attempt in range(3):
        try:
            if stream:
                out = ""
                for chunk in client.stream(messages):
                    out += chunk.content
                    print(chunk.content, end="", flush=True)
                print()
                return strip_think(out)
            return strip_think(client.invoke(messages).content)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def extract_json(text: str):
    cleaned = strip_think(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    candidates = [fence.group(1), cleaned] if fence else [cleaned]
    decoder = json.JSONDecoder()
    for cand in candidates:
        for m in re.finditer(r"[\{\[]", cand):
            try:
                obj, _ = decoder.raw_decode(cand[m.start():])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def language_of(filename: str) -> Optional[str]:
    if filename in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[filename]
    return EXT_LANG.get(Path(filename).suffix.lower())


def safe_rel_path(rel) -> Optional[str]:
    rel = str(rel or "").strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if not rel or rel.startswith("/") or ".." in rel.split("/"):
        return None
    if re.match(r"^[A-Za-z]:", rel):
        return None
    return rel


def resolve_in_project(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes project root: {rel}")
    return target


def read_text_lf(path: Path) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), newline


def write_text_nl(path: Path, text: str, newline: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", newline))


SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:"
    r"(?:def|class|function|func|fn|interface|struct|enum|trait|impl|module|namespace)\b"
    r"|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\(|function)"
    r"|(?:public|private|protected|static|internal)\b.*\()"
)


def symbol_outline(content: str, limit: int = 60) -> str:
    out = []
    for n, line in enumerate(content.split("\n"), 1):
        if SYMBOL_RE.match(line):
            out.append(f"{n}: {line.strip()[:100]}")
            if len(out) >= limit:
                break
    return "\n".join(out)


def focus_excerpt(content: str, function_name: Optional[str], max_chars: int) -> Tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    lines = content.split("\n")
    idx = 0
    name = ""
    if function_name:
        name = function_name.split("(")[0].split(".")[-1].strip()
    if name:
        token = re.compile(r"\b" + re.escape(name) + r"\b")
        for i, line in enumerate(lines):
            if token.search(line):
                idx = i
                break
    start = max(0, idx - 25)
    out, size = [], 0
    for line in lines[start:]:
        if size + len(line) + 1 > max_chars:
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out), True


def manifest_brief(manifest: dict) -> str:
    brief = {
        "primary_language": manifest.get("primary_language"),
        "frameworks": manifest.get("frameworks"),
        "architecture": manifest.get("architecture"),
        "conventions": manifest.get("conventions"),
        "entry_points": manifest.get("entry_points"),
    }
    return json.dumps(brief, indent=2)[:1800]


def load_manifest(state) -> dict:
    manifest = state.get("manifest") or {}
    if manifest:
        return manifest
    path = state["run_dir"] / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def language_for_file(manifest: dict, rel: str) -> str:
    ext = Path(rel).suffix.lower() or Path(rel).name
    ext_map = manifest.get("extension_language_map", {})
    return ext_map.get(ext) or language_of(Path(rel).name) or manifest.get("primary_language") or "Unknown"


# ==========================================
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    project_address: str
    user_request: str
    plan_stack: list
    planned_tasks: list
    issue_report: str
    run_dir: Path
    manifest: dict
    current_task: dict
    edit_result: dict
    task_results: list
    tool_trace: list
    review_attempts: int
    iterations: int
    task_attempts: Annotated[dict, operator.ior]
    total_tasks: int
    start_time: float


# ==========================================
# 3. NODES
# ==========================================
def requirement_node(state: AgentState):
    print("🔍 [Requirement] Ingesting project address and request...")
    address = str(state.get("project_address", "")).strip().strip('"').strip("'")
    request = str(state.get("user_request", "")).strip()
    if not request:
        raise ValueError("user_request is empty.")
    root = Path(address).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project_address is not a directory: {root}")

    write_log(state["run_dir"], "Requirement Node", f"Project: {root}\nRequest: {request}")
    (state["run_dir"] / "request.md").write_text(
        f"# User Request\n\n{request}\n\n**Project:** `{root}`\n", encoding="utf-8"
    )
    return {
        "project_address": str(root),
        "user_request": request,
        "plan_stack": [],
        "planned_tasks": [],
        "issue_report": "",
        "manifest": {},
        "current_task": {},
        "edit_result": {},
        "task_results": [],
        "tool_trace": [],
        "review_attempts": 0,
        "iterations": 0,
        "task_attempts": {},
        "total_tasks": 0,
        "start_time": time.perf_counter(),
    }


def scan_project(root: Path) -> dict:
    files: List[str] = []
    marker_files: List[str] = []
    lang_stats: dict = {}
    ext_map: dict = {}
    full = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if len(files) >= MAX_SCAN_FILES:
                full = True
                break
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            files.append(rel)
            if name in FRAMEWORK_SIGNATURES or name.endswith(".csproj"):
                marker_files.append(rel)
            lang = language_of(name)
            if not lang:
                continue
            ext_map[path.suffix.lower() or name] = lang
            stat = lang_stats.setdefault(lang, {"files": 0, "lines": 0})
            stat["files"] += 1
            try:
                if path.stat().st_size <= 512 * 1024:
                    with open(path, "rb") as f:
                        stat["lines"] += f.read().count(b"\n") + 1
            except OSError:
                pass
        if full:
            break
    return {"files": files, "marker_files": marker_files, "lang_stats": lang_stats, "ext_map": ext_map}


def detect_frameworks(root: Path, marker_files: List[str]) -> List[str]:
    found = []
    for rel in marker_files[:40]:
        name = Path(rel).name
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="ignore")[:200_000]
        except OSError:
            continue
        if name == "package.json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict):
                deps = {}
                for section in ("dependencies", "devDependencies"):
                    if isinstance(data.get(section), dict):
                        deps.update(data[section])
                for key, label in FRAMEWORK_SIGNATURES["package.json"].items():
                    if key in deps:
                        found.append(label)
        elif name.endswith(".csproj"):
            found.append(".NET")
            if "Microsoft.NET.Sdk.Web" in text:
                found.append("ASP.NET Core")
        else:
            low = text.lower()
            for key, label in FRAMEWORK_SIGNATURES.get(name, {}).items():
                if key in low:
                    found.append(label)
    return sorted(set(found))


def analyzer_node(state: AgentState):
    print("\n🧭 [Analyzer] Scanning project and building manifest.json...")
    root = Path(state["project_address"])
    scan = scan_project(root)
    files = scan["files"]
    frameworks = detect_frameworks(root, scan["marker_files"])

    lang_stats = dict(sorted(scan["lang_stats"].items(), key=lambda kv: kv[1]["lines"], reverse=True))
    code_langs = {k: v for k, v in lang_stats.items() if k not in DATA_LANGS}
    ranking = code_langs or lang_stats
    primary_language = next(iter(ranking), "Unknown")

    entry_points = [f for f in files if Path(f).name in ENTRY_POINT_NAMES][:10]
    top_dirs = sorted({f.split("/")[0] for f in files if "/" in f})

    readme_head = ""
    for f in files:
        if "/" not in f and f.lower().startswith("readme"):
            try:
                readme_head = (root / f).read_text(encoding="utf-8", errors="ignore")[:1500]
            except OSError:
                pass
            break

    system = (
        "You are a software architect analysing an unfamiliar repository. "
        "Reply with a single JSON object only."
    )
    human = (
        f"Languages (files/lines): {json.dumps(lang_stats)}\n"
        f"Detected frameworks: {frameworks}\n"
        f"Entry point candidates: {entry_points}\n"
        f"Top-level directories: {top_dirs}\n"
        f"File tree (first 150 of {len(files)}):\n" + "\n".join(files[:150]) + "\n\n"
        f"README (head):\n{readme_head}\n\n"
        'Return JSON: {"architecture": "2-4 sentences: layers, modules, data flow, patterns", '
        '"entry_points": ["relative/path", ...], '
        '"conventions": "naming, testing, style and structure conventions you can infer"}'
    )
    data = extract_json(call_llm(llm_json, system, human))
    if not isinstance(data, dict):
        data = {}

    file_set = set(files)
    llm_entries = data.get("entry_points")
    if isinstance(llm_entries, list):
        llm_entries = [e for e in llm_entries if isinstance(e, str) and e in file_set]
    else:
        llm_entries = []

    manifest = {
        "project_address": str(root),
        "project_name": root.name,
        "primary_language": primary_language,
        "languages": lang_stats,
        "extension_language_map": scan["ext_map"],
        "frameworks": frameworks,
        "build_files": scan["marker_files"],
        "entry_points": llm_entries or entry_points,
        "architecture": str(data.get("architecture") or (
            f"{primary_language} project with top-level modules: {', '.join(top_dirs) or 'flat layout'}."
        )),
        "conventions": str(data.get("conventions") or "Follow the existing style of each file."),
        "total_files": len(files),
        "file_tree": files[:MAX_TREE_ENTRIES],
    }
    (state["run_dir"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_log(state["run_dir"], "Analyzer Node - MANIFEST", json.dumps(manifest, indent=2)[:6000])
    print(f"   Primary language: {primary_language} | Frameworks: {', '.join(frameworks) or 'none detected'}")
    trace_entry = {
        "step": "analyzer",
        "tool": "scan_project_and_detect_frameworks",
        "arguments": {"project_address": str(root), "max_scan_files": MAX_SCAN_FILES},
        "status": "success",
        "detail": f"Scanned {len(files)} files, primary={primary_language}",
    }
    return {"manifest": manifest, "tool_trace": list(state.get("tool_trace", [])) + [trace_entry]}


def rank_files(tree: List[str], request: str) -> List[str]:
    tokens = set(re.findall(r"[a-zA-Z_]{3,}", request.lower()))

    def score(path: str) -> float:
        low = path.lower()
        name_low = Path(path).stem.lower()
        exact_stem = sum(3 for t in tokens if t == name_low)
        substr = sum(1 for t in tokens if t in low)
        return -(exact_stem + substr)

    return sorted(tree, key=score)


def select_relevant_files(state: AgentState, manifest: dict) -> List[str]:
    tree = [f for f in manifest.get("file_tree", []) if language_of(Path(f).name)]
    if len(tree) <= 8:
        return tree
    candidates = rank_files(tree, state["user_request"])[:200]
    system = "You are a senior engineer choosing which files must be read to implement a change. Reply with JSON only."
    human = (
        f"USER REQUEST:\n{state['user_request']}\n\n"
        f"PROJECT:\n{manifest_brief(manifest)}\n\n"
        "CANDIDATE FILES:\n" + "\n".join(candidates) + "\n\n"
        'Return JSON: {"files": ["relative/path", ...]} with at most 6 files that must be read or edited. '
        "Use only paths from the list."
    )
    data = extract_json(call_llm(llm_json, system, human))
    picked = []
    if isinstance(data, dict) and isinstance(data.get("files"), list):
        tree_set = set(tree)
        picked = [f for f in data["files"] if isinstance(f, str) and f in tree_set][:6]
    if not picked:
        ranked_entries = [f for f in manifest.get("entry_points", []) if f in set(tree)]
        picked = (candidates[:4] + ranked_entries)[:6]
    return list(dict.fromkeys(picked))


def normalize_plan(raw_tasks, root: Path, tree_set: set) -> List[dict]:
    plan: List[dict] = []
    planned_new: set = set()
    seen_signatures: set = set()
    if not isinstance(raw_tasks, list):
        return plan
    for raw in raw_tasks:
        if not isinstance(raw, dict):
            continue
        rel = safe_rel_path(raw.get("file"))
        if not rel:
            continue
        description = str(raw.get("description") or "").strip()
        if not description:
            continue
        exists = (root / rel).is_file() or rel in tree_set
        action = str(raw.get("action") or "edit").strip().lower()
        if action not in ("edit", "create"):
            action = "edit"
        if action == "edit" and not exists and rel not in planned_new:
            action = "create"
        elif action == "create" and (exists or rel in planned_new):
            action = "edit"
        if action == "create":
            planned_new.add(rel)
        func = raw.get("function")
        func = str(func).strip() if func and str(func).strip().lower() not in ("null", "none") else None
        sig = (action, rel, func, description.lower())
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        plan.append({"action": action, "file": rel, "function": func, "description": description})
        if len(plan) >= MAX_TASKS:
            break
    for i, task in enumerate(plan, 1):
        task["step"] = i
        task["id"] = f"T{i:02d}"
    return plan


def planner_node(state: AgentState):
    print("\n🗺️  [Planner] Building LIFO execution stack...")
    root = Path(state["project_address"]).resolve()
    manifest = load_manifest(state)
    tree_set = set(manifest.get("file_tree", []))

    selected = select_relevant_files(state, manifest)
    per_file = max(1500, PLANNER_CONTENT_BUDGET // max(len(selected), 1))
    blocks = []
    for rel in selected:
        try:
            content, _ = read_text_lf(root / rel)
        except (OSError, UnicodeDecodeError):
            continue
        if len(content) > per_file:
            body = content[:per_file] + "\n[... truncated ...]\nSYMBOL OUTLINE (line: signature):\n" + symbol_outline(content)
        else:
            body = content
        blocks.append(f"### {rel}\n{body}")
    files_block = "\n\n".join(blocks) or "(project has no readable source files yet)"

    system = (
        "You are a principal engineer who turns a change request into a granular, ordered execution plan "
        "for a code-editing agent. Reply with a single JSON object only."
    )
    base_human = (
        f"USER REQUEST:\n{state['user_request']}\n\n"
        f"PROJECT MANIFEST:\n{manifest_brief(manifest)}\n\n"
        f"RELEVANT FILES:\n{files_block}\n\n"
        'Return JSON: {"tasks": [{"file": "relative/path", "action": "edit" | "create", '
        '"function": "function/class/section to change, or null", "description": "the exact change to make"}]}\n'
        "Rules:\n"
        f"- List tasks in EXACT execution order (first item is executed first). At most {MAX_TASKS} tasks.\n"
        "- One task = one file and one function/logical unit. Be granular and specific.\n"
        '- Use "edit" for existing files and "create" only for files that do not exist yet.\n'
        "- Only touch files necessary for the request; use relative paths from the project root.\n"
        "- Later tasks may rely on files created or symbols added by earlier tasks."
    )
    plan: List[dict] = []
    for attempt in range(2):
        human = base_human
        if attempt:
            human += "\n\nYour previous answer contained no valid tasks. Return at least one task with file, action and description."
        data = extract_json(call_llm(llm_json, system, human))
        raw_tasks = data.get("tasks") if isinstance(data, dict) else data
        plan = normalize_plan(raw_tasks, root, tree_set)
        if plan:
            break
    if not plan:
        raise RuntimeError("Planner produced no executable tasks for this request.")

    plan_stack = list(reversed(plan))
    (state["run_dir"] / "plan.json").write_text(
        json.dumps({"execution_order": plan, "stack_bottom_to_top": plan_stack}, indent=2), encoding="utf-8"
    )
    write_log(state["run_dir"], "Planner Node - PLAN", json.dumps(plan, indent=2))
    for t in plan:
        print(f"   {t['id']} [{t['action']}] {t['file']} :: {t['function'] or '-'} — {t['description'][:90]}")
    trace_entry = {
        "step": "planner",
        "tool": "generate_execution_plan",
        "arguments": {"selected_files": selected, "task_count": len(plan)},
        "status": "success",
        "detail": f"Created {len(plan)} tasks in LIFO stack",
    }
    return {
        "plan_stack": plan_stack,
        "planned_tasks": plan,
        "total_tasks": len(plan),
        "tool_trace": list(state.get("tool_trace", [])) + [trace_entry],
    }


# ---------- 3.4 Editor ----------
BLOCK_RE = re.compile(
    r"<<<<<<<+\s*SEARCH[ \t]*\r?\n(.*?)\r?\n=======+[ \t]*\r?\n(.*?)\r?\n?>>>>>>>+\s*REPLACE",
    re.DOTALL,
)


def parse_search_replace_blocks(raw: str) -> List[Tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in BLOCK_RE.finditer(raw)]


def sanitize_code_output(raw: str) -> str:
    fence = re.search(r"```[\w+#.-]*\n(.*?)\n?```", raw, re.DOTALL)
    return (fence.group(1) if fence else raw).strip("\n")


def apply_search_replace(content: str, search: str, replace: str) -> Tuple[Optional[str], str]:
    count = content.count(search)
    if count == 1:
        return content.replace(search, replace, 1), "exact match"
    if count > 1:
        return None, f"SEARCH block matches {count} locations; add more surrounding lines to make it unique."

    c_lines = content.split("\n")
    s_lines = search.split("\n")
    while s_lines and not s_lines[0].strip():
        s_lines.pop(0)
    while s_lines and not s_lines[-1].strip():
        s_lines.pop()
    if not s_lines:
        return None, "SEARCH block is blank."
    s_norm = [l.strip() for l in s_lines]
    hits = [
        i for i in range(len(c_lines) - len(s_lines) + 1)
        if all(c_lines[i + j].strip() == s_norm[j] for j in range(len(s_lines)))
    ]
    if len(hits) == 1:
        i = hits[0]
        file_indent = re.match(r"\s*", c_lines[i]).group(0)
        search_indent = re.match(r"\s*", s_lines[0]).group(0)
        new_lines = []
        for line in replace.split("\n"):
            if line.strip() and line.startswith(search_indent):
                new_lines.append(file_indent + line[len(search_indent):])
            else:
                new_lines.append(line)
        result = c_lines[:i] + new_lines + c_lines[i + len(s_lines):]
        return "\n".join(result), "whitespace-tolerant match"
    if len(hits) > 1:
        return None, f"SEARCH block matches {len(hits)} locations after whitespace normalisation; add more context."

    # Fuzzy similarity fallback for minor whitespace or comment drift
    window = len(s_lines)
    best_ratio, best_idx = 0.0, -1
    target_block = "\n".join(s_norm)
    for i in range(max(0, len(c_lines) - window + 1)):
        cand_block = "\n".join(l.strip() for l in c_lines[i:i + window])
        ratio = difflib.SequenceMatcher(None, target_block, cand_block).ratio()
        if ratio > best_ratio:
            best_ratio, best_idx = ratio, i
    if best_ratio >= 0.92 and best_idx >= 0:
        file_indent = re.match(r"\s*", c_lines[best_idx]).group(0)
        search_indent = re.match(r"\s*", s_lines[0]).group(0)
        new_lines = [
            (file_indent + line[len(search_indent):]) if (line.strip() and line.startswith(search_indent)) else line
            for line in replace.split("\n")
        ]
        result = c_lines[:best_idx] + new_lines + c_lines[best_idx + window:]
        return "\n".join(result), f"fuzzy-anchored match ({best_ratio:.2f})"

    return None, "SEARCH block was not found in the file (text must be copied verbatim)."


def make_diff(before: str, after: str, rel: str) -> str:
    diff = "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="", n=3
    ))
    return diff[:MAX_DIFF_CHARS] + ("\n[... diff truncated ...]" if len(diff) > MAX_DIFF_CHARS else "")


def build_editor_system_prompt(language: str, manifest: dict, action: str) -> str:
    guide = LANGUAGE_GUIDES.get(language, DEFAULT_LANGUAGE_GUIDE)
    frameworks = ", ".join(manifest.get("frameworks") or []) or "none detected"
    fmt = CREATE_FORMAT if action == "create" else EDIT_FORMAT
    return (
        f"You are a senior {language} engineer making precise, minimal code changes in an existing repository.\n"
        f"Project frameworks: {frameworks}.\n"
        f"Architecture: {manifest.get('architecture', 'unknown')}\n"
        f"Conventions: {manifest.get('conventions', 'follow existing style')}\n"
        f"{language} rules: {guide}\n"
        "Never leave TODOs, placeholders or pseudo-code. Every change must be complete, compilable/runnable and consistent with the rest of the codebase.\n\n"
        f"{fmt}"
    )


def backup_file(run_dir: Path, task: dict, target: Path) -> str:
    if not target.exists():
        return ""
    dest = run_dir / "backups" / task["id"] / (task["file"] + ".bak")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, dest)
    return str(dest)


def _failed(task: dict, attempt: int, detail: str, tool_used: str = "unknown", args_summary: Optional[dict] = None) -> dict:
    return {
        "task_id": task["id"],
        "file": task["file"],
        "status": "failed",
        "detail": detail,
        "diff": "",
        "attempt": attempt,
        "tool": tool_used,
        "arguments": args_summary or {"file": task["file"], "function": task.get("function")},
    }


def editor_node(state: AgentState):
    root = Path(state["project_address"]).resolve()
    run_dir = state["run_dir"]
    manifest = load_manifest(state)
    plan_stack = list(state.get("plan_stack", []))
    task = dict(state.get("current_task") or {})
    issue = state.get("issue_report", "")
    retrying = bool(task) and bool(issue.strip())

    if not retrying:
        if not plan_stack:
            raise RuntimeError("Editor invoked with an empty plan_stack.")
        task = dict(plan_stack.pop())
        task["started_at"] = time.perf_counter()
        target = resolve_in_project(root, task["file"])
        task["existed_before"] = target.exists()
        task["backup_path"] = backup_file(run_dir, task, target)
        issue = ""
        review_attempts = 0
    else:
        target = resolve_in_project(root, task["file"])
        review_attempts = state.get("review_attempts", 0)

    attempt = state.get("task_attempts", {}).get(task["id"], 0) + 1
    language = language_for_file(manifest, task["file"])
    action = task["action"]
    if action == "create" and target.is_file() and task.get("existed_before"):
        action = "edit"
        task["action"] = "edit"
    elif action == "edit" and not target.is_file():
        action = "create"
        task["action"] = "create"

    print(f"\n💻 [Editor] {task['id']} ({action}, {language}) → {task['file']} | attempt {attempt}")
    if task.get("backup_path"):
        print(f"   🗄️  Backup: {task['backup_path']}")

    system = build_editor_system_prompt(language, manifest, action)
    issue_block = f"\nREVIEWER ISSUE REPORT (fix exactly these problems):\n{issue}\n" if issue.strip() else ""
    header = (
        f"OVERALL GOAL: {state['user_request']}\n\n"
        f"CURRENT TASK {task['id']}: {action} `{task['file']}` — function/section: {task.get('function') or 'n/a'}\n"
        f"WHAT TO DO: {task['description']}\n"
        f"{issue_block}"
    )

    if action == "edit":
        if not target.is_file():
            result = _failed(task, attempt, f"Target file does not exist: {task['file']}", "search_replace_edit")
        else:
            try:
                before, newline = read_text_lf(target)
            except (OSError, UnicodeDecodeError) as e:
                before, newline = None, "\n"
                result = _failed(task, attempt, f"Cannot read target file as UTF-8 text: {e}", "search_replace_edit")
            if before is not None:
                excerpt, partial = focus_excerpt(before, task.get("function"), MAX_FILE_CHARS)
                human = (
                    header
                    + ("\nThe file already contains your previous edit.\n" if retrying else "")
                    + f"\nCURRENT CONTENT OF `{task['file']}`{' (EXCERPT — file is longer)' if partial else ''}:\n<file>\n{excerpt}\n</file>\n\n"
                    "Return only SEARCH/REPLACE blocks."
                )
                write_log(run_dir, f"Editor Node [{task['id']}] - PROMPT", system + "\n\n" + human)
                raw = call_llm(llm_editor, system, human, stream=True)
                write_log(run_dir, f"Editor Node [{task['id']}] - RAW OUTPUT", raw)
                result = apply_edit_blocks(task, attempt, target, before, newline, raw)
    else:
        previous = ""
        newline = "\n"
        if target.exists():
            previous, newline = read_text_lf(target)
        human = header
        if previous:
            human += f"\nCURRENT CONTENT OF `{task['file']}` (created by you earlier; fix it):\n<file>\n{previous[:MAX_FILE_CHARS]}\n</file>\n"
        human += f"\nCreate `{task['file']}` in full."
        write_log(run_dir, f"Editor Node [{task['id']}] - PROMPT", system + "\n\n" + human)
        raw = call_llm(llm_editor, system, human, stream=True)
        write_log(run_dir, f"Editor Node [{task['id']}] - RAW OUTPUT", raw)
        content = sanitize_code_output(raw)
        if len(content.strip()) < 1:
            result = _failed(task, attempt, "Model returned no file content.", "create_file")
        else:
            write_text_nl(target, content + "\n", newline)
            result = {
                "task_id": task["id"],
                "file": task["file"],
                "status": "applied",
                "detail": "new file written" if not previous else "new file rewritten",
                "diff": make_diff(previous, content + "\n", task["file"]),
                "attempt": attempt,
                "tool": "create_file",
                "arguments": {
                    "file": task["file"],
                    "function": task.get("function"),
                    "chars_written": len(content),
                },
            }

    write_log(run_dir, f"Editor Node [{task['id']}] - RESULT", f"{result['status']}: {result['detail']}\n{result['diff']}")
    print(f"   → {result['status']}: {result['detail']}")
    trace_entry = {
        "step": f"editor_{task['id']}_attempt_{attempt}",
        "tool": result.get("tool", action),
        "arguments": result.get("arguments", {"file": task["file"], "action": action}),
        "status": result["status"],
        "detail": result["detail"],
    }
    return {
        "plan_stack": plan_stack,
        "current_task": task,
        "edit_result": result,
        "review_attempts": review_attempts,
        "iterations": state.get("iterations", 0) + 1,
        "task_attempts": {task["id"]: attempt},
        "tool_trace": list(state.get("tool_trace", [])) + [trace_entry],
    }


def apply_edit_blocks(task: dict, attempt: int, target: Path, before: str, newline: str, raw: str) -> dict:
    blocks = parse_search_replace_blocks(raw)
    args_meta = {
        "file": task["file"],
        "function": task.get("function"),
        "blocks_parsed": len(blocks),
    }
    if not blocks:
        return _failed(task, attempt, "No valid SEARCH/REPLACE blocks were found in the model output.", "search_replace_edit", args_meta)
    after = before
    total_lines = len(before.splitlines())
    notes = []
    for n, (search, replace) in enumerate(blocks, 1):
        if not search.strip():
            return _failed(task, attempt, f"Block {n}: SEARCH is empty; edits must anchor on existing code.", "search_replace_edit", args_meta)
        if total_lines > 15 and len(search) > 0.85 * len(before):
            return _failed(task, attempt, f"Block {n}: SEARCH spans nearly the whole file; whole-file rewrites are forbidden. Use targeted blocks.", "search_replace_edit", args_meta)
        updated, note = apply_search_replace(after, search, replace)
        if updated is None:
            return _failed(task, attempt, f"Block {n}: {note}", "search_replace_edit", args_meta)
        after = updated
        notes.append(f"block {n}: {note}")
    if after == before:
        return _failed(task, attempt, "The edit produced no change to the file.", "search_replace_edit", args_meta)
    backup_guard = target.with_suffix(target.suffix + ".tmp_edit")
    try:
        write_text_nl(backup_guard, after, newline)
        os.replace(backup_guard, target)
    finally:
        if backup_guard.exists():
            backup_guard.unlink()
    args_meta["match_notes"] = notes
    return {
        "task_id": task["id"],
        "file": task["file"],
        "status": "applied",
        "detail": f"{len(blocks)} block(s) applied ({'; '.join(notes)})",
        "diff": make_diff(before, after, task["file"]),
        "attempt": attempt,
        "tool": "search_replace_edit",
        "arguments": args_meta,
    }


# ---------- 3.5 Reviewer ----------
def static_check(path: Path) -> Tuple[bool, str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            return True, "python syntax OK"
        if suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            return True, "json syntax OK"
        if suffix in (".js", ".mjs", ".cjs") and shutil.which("node"):
            r = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=30)
            return r.returncode == 0, (r.stderr.strip()[:1500] or "javascript syntax OK")
        if suffix == ".php" and shutil.which("php"):
            r = subprocess.run(["php", "-l", str(path)], capture_output=True, text=True, timeout=30)
            return r.returncode == 0, (r.stderr.strip()[:1500] or r.stdout.strip()[:1500] or "php syntax OK")
        if suffix in (".sh", ".bash") and shutil.which("bash"):
            r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=30)
            return r.returncode == 0, (r.stderr.strip()[:1500] or "shell syntax OK")
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except json.JSONDecodeError as e:
        return False, f"JSONDecodeError: {e.msg} (line {e.lineno})"
    except ValueError as e:
        return False, f"Invalid source: {e}"
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError) as e:
        return True, f"static check skipped: {e}"
    return True, "no static checker available for this file type"


def restore_from_backup(task: dict, target: Path):
    if task.get("existed_before") and task.get("backup_path"):
        shutil.copy2(task["backup_path"], target)
    elif not task.get("existed_before"):
        target.unlink(missing_ok=True)


def build_issue_md(task: dict, attempt: int, body: str) -> str:
    return (
        "# Issue Report\n\n"
        f"- **Task:** {task['id']} ({task['action']})\n"
        f"- **File:** `{task['file']}`\n"
        f"- **Function/section:** {task.get('function') or 'n/a'}\n"
        f"- **Attempt:** {attempt}/{MAX_REVIEW_ATTEMPTS}\n\n"
        f"## Findings\n\n{body.strip()}\n"
    )


REVIEWER_SYSTEM = (
    "You are a strict senior code reviewer. You verify that one specific edit fulfils its task, "
    "keeps the code correct and consistent with the project, and does not damage unrelated code."
)


def reviewer_node(state: AgentState):
    run_dir = state["run_dir"]
    root = Path(state["project_address"]).resolve()
    task = state["current_task"]
    result = state["edit_result"]
    manifest = load_manifest(state)
    target = resolve_in_project(root, task["file"])
    attempt = state.get("review_attempts", 0) + 1
    print(f"\n🔎 [Reviewer] Auditing {task['id']} → {task['file']} (attempt {attempt}/{MAX_REVIEW_ATTEMPTS})")

    passed = False
    body = ""
    check_msg = "not run"
    if result.get("status") != "applied":
        body = "The Editor's output could not be applied.\n\n" + result.get("detail", "")
    else:
        ok, check_msg = static_check(target)
        if not ok:
            body = f"Static syntax check failed:\n\n```\n{check_msg}\n```"
        else:
            try:
                current, _ = read_text_lf(target)
            except (OSError, UnicodeDecodeError) as e:
                current = f"<unreadable: {e}>"
            excerpt, partial = focus_excerpt(current, task.get("function"), MAX_FILE_CHARS)
            human = (
                f"OVERALL GOAL:\n{state['user_request']}\n\n"
                f"PROJECT MANIFEST:\n{manifest_brief(manifest)}\n\n"
                f"TASK UNDER REVIEW ({task['id']}): {task['action']} `{task['file']}` — "
                f"function/section: {task.get('function') or 'n/a'}\n"
                f"TASK DESCRIPTION: {task['description']}\n\n"
                f"STATIC CHECK: {check_msg}\n\n"
                f"DIFF APPLIED BY THE EDITOR:\n```diff\n{result.get('diff', '')}\n```\n\n"
                f"FILE AFTER EDIT{' (EXCERPT)' if partial else ''}:\n<file>\n{excerpt}\n</file>\n\n"
                "Check: (1) the diff fully accomplishes the task description; (2) syntax and logic are correct; "
                "(3) names, imports, signatures and conventions match the manifest and surrounding code; "
                "(4) nothing unrelated was changed or removed; (5) no TODOs, stubs or placeholders.\n\n"
                "Answer in this exact format:\n"
                "VERDICT: PASS\n"
                "or\n"
                "VERDICT: FAIL\n"
                "## Issue\n<what is wrong, citing the exact lines>\n## Required Fix\n<precise change needed>"
            )
            review = call_llm(llm, REVIEWER_SYSTEM, human)
            write_log(run_dir, f"Reviewer Node [{task['id']}]", review)
            m = re.search(r"VERDICT:\s*(PASS|FAIL)", review, re.IGNORECASE)
            if m:
                passed = m.group(1).upper() == "PASS"
            else:
                passed = bool(re.match(r"^\s*PASS\b", review, re.IGNORECASE))
            if not passed:
                body = re.sub(r"(?is)^.*?VERDICT:\s*FAIL\s*", "", review, count=1).strip() or review

    duration = round(time.perf_counter() - task.get("started_at", time.perf_counter()), 2)
    attempts_used = state.get("task_attempts", {}).get(task["id"], attempt)
    record = {
        "id": task["id"],
        "file": task["file"],
        "function": task.get("function"),
        "description": task.get("description"),
        "action": task["action"],
        "tool": result.get("tool", task["action"]),
        "arguments": result.get("arguments", {}),
        "diff": result.get("diff", "")[:2000],
        "static_check": check_msg,
        "attempts": attempts_used,
        "duration_sec": duration,
    }
    issue_path = run_dir / "issue.md"
    trace_entry = {
        "step": f"reviewer_{task['id']}_attempt_{attempt}",
        "tool": "static_and_llm_review",
        "arguments": {"file": task["file"], "static_check": check_msg},
        "status": "passed" if passed else "failed",
        "detail": "PASS" if passed else body[:240],
    }
    updated_trace = list(state.get("tool_trace", [])) + [trace_entry]

    if passed:
        print(f"   ✅ {task['id']} passed review.")
        issue_path.unlink(missing_ok=True)
        write_log(run_dir, f"Reviewer Node [{task['id']}]", "PASS")
        return {
            "issue_report": "",
            "current_task": {},
            "edit_result": {},
            "review_attempts": 0,
            "task_results": list(state.get("task_results", [])) + [{**record, "status": "passed"}],
            "tool_trace": updated_trace,
        }

    issue_md = build_issue_md(task, attempt, body)
    issue_path.write_text(issue_md, encoding="utf-8")
    write_log(run_dir, f"Reviewer Node [{task['id']}] - ISSUE", issue_md)

    if attempt >= MAX_REVIEW_ATTEMPTS:
        print(f"   ⚠️ {task['id']} failed {MAX_REVIEW_ATTEMPTS} reviews — restoring backup and moving on.")
        restore_from_backup(task, target)
        write_log(run_dir, f"Reviewer Node [{task['id']}]", "MAX ATTEMPTS REACHED — backup restored, task marked failed")
        return {
            "issue_report": "",
            "current_task": {},
            "edit_result": {},
            "review_attempts": 0,
            "task_results": list(state.get("task_results", [])) + [{**record, "status": "failed"}],
            "tool_trace": updated_trace,
        }

    print(f"   ❌ Issues found in {task['id']}. Routing back to Editor.")
    return {"issue_report": issue_md, "review_attempts": attempt, "tool_trace": updated_trace}


# ==========================================
# 4. EVALUATION (DeepEval + Agentic Matrix)
# ==========================================
if DEEPEVAL_AVAILABLE:
    class OllamaEvalModel(DeepEvalBaseLLM):
        def __init__(self, chat_model, name):
            self._chat = chat_model
            self._name = name

        def load_model(self):
            return self._chat

        def generate(self, prompt: str) -> str:
            return strip_think(self._chat.invoke(prompt).content)

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return self._name


def classify_metric_score(metric_name: str, score: Optional[float]) -> dict:
    spec = AGENTIC_METRIC_SPECS[metric_name]
    if score is None:
        return {
            "score": None,
            "threshold": spec["threshold"],
            "passed": False,
            "band": "N/A",
            "bad_range": spec["bad_label"],
            "good_range": spec["good_label"],
        }
    s = round(float(score), 2)
    passed = s >= spec["threshold"]
    if s <= spec["bad_max"]:
        band = "Bad"
    elif s >= spec["good_min"]:
        band = "Good"
    else:
        band = "Moderate"
    return {
        "score": s,
        "threshold": spec["threshold"],
        "passed": passed,
        "band": band,
        "bad_range": spec["bad_label"],
        "good_range": spec["good_label"],
    }


def compute_deterministic_agentic_fallbacks(
    planned_tasks: list, results: list, tool_trace: list, iterations: int
) -> dict:
    total_planned = max(len(planned_tasks), 1)
    valid_plan_tasks = sum(
        1 for t in planned_tasks
        if t.get("file") and t.get("action") in ("edit", "create") and len(t.get("description", "")) >= 10
    )
    plan_quality = round(min(1.0, valid_plan_tasks / total_planned), 2)

    planned_ids = {t["id"]: t for t in planned_tasks}
    adhered = sum(
        1 for r in results
        if r["id"] in planned_ids and r["file"] == planned_ids[r["id"]]["file"] and r["status"] == "passed"
    )
    plan_adherence = round(adhered / total_planned, 2)

    editor_traces = [t for t in tool_trace if str(t.get("step", "")).startswith("editor_")]
    if editor_traces:
        correct_tools = sum(
            1 for t in editor_traces
            if t.get("tool") in ("search_replace_edit", "create_file") and t.get("status") == "applied"
        )
        tool_correctness = round(correct_tools / len(editor_traces), 2)
    else:
        tool_correctness = 0.0

    if editor_traces:
        valid_args = sum(
            1 for t in editor_traces
            if t.get("status") == "applied" and isinstance(t.get("arguments"), dict) and t["arguments"].get("file")
        )
        argument_correctness = round(valid_args / len(editor_traces), 2)
    else:
        argument_correctness = 0.0

    total_attempts = sum(r.get("attempts", 1) for r in results) or max(iterations, 1)
    step_efficiency = round(max(0.0, min(1.0, total_planned / max(total_attempts, total_planned))), 2)

    passed_tasks = sum(1 for r in results if r.get("status") == "passed")
    task_completion = round(passed_tasks / total_planned, 2)

    return {
        "Plan Quality": (plan_quality, "Deterministic fallback: ratio of well-formed, scoped tasks in plan."),
        "Plan Adherence": (plan_adherence, "Deterministic fallback: ratio of planned tasks executed on target files and passed."),
        "Tool Correctness": (tool_correctness, "Deterministic fallback: ratio of editor invocations selecting & applying valid file tools."),
        "Argument Correctness": (argument_correctness, "Deterministic fallback: ratio of tool calls with valid file/block arguments."),
        "Step Efficiency": (step_efficiency, f"Deterministic fallback: {total_planned} planned task(s) completed in {total_attempts} attempt(s)."),
        "Task Completion": (task_completion, f"Deterministic fallback: {passed_tasks}/{total_planned} tasks passed review."),
    }


def _build_file_deepeval_metrics() -> list:
    judge = OllamaEvalModel(llm, llm.model)
    return [
        GEval(
            name="Correctness",
            criteria=(
                "The ACTUAL_OUTPUT is a source file after an automated edit. Using the user request, manifest and plan "
                "given in INPUT as ground truth: do all referenced identifiers, imports, ids and function calls exist "
                "and match the manifest/project conventions? Does each changed function's logic plausibly fulfil the "
                "request? Flag any obvious runtime or syntax bug."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.70,
        ),
        GEval(
            name="Completeness",
            criteria=(
                "Using the user request and plan given in INPUT as the spec, does ACTUAL_OUTPUT implement "
                "every requested change for this file with real logic, not a TODO or empty stub?"
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.70,
        ),
    ]


def _build_agentic_deepeval_metrics() -> dict:
    judge = OllamaEvalModel(llm, llm.model)
    metrics = {}
    for name, spec in AGENTIC_METRIC_SPECS.items():
        metrics[name] = GEval(
            name=name,
            criteria=spec["criteria"],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=spec["threshold"],
        )
    return metrics


def evaluate_agentic_workflow(
    state: AgentState, manifest: dict, planned_tasks: list, results: list, tool_trace: list
) -> dict:
    fallbacks = compute_deterministic_agentic_fallbacks(
        planned_tasks, results, tool_trace, state.get("iterations", len(results))
    )
    input_payload = json.dumps(
        {
            "user_request": state.get("user_request", ""),
            "manifest_summary": {
                "primary_language": manifest.get("primary_language"),
                "frameworks": manifest.get("frameworks"),
                "architecture": manifest.get("architecture"),
            },
            "planned_tasks": planned_tasks,
        },
        indent=2,
    )[:5000]

    actual_payload = json.dumps(
        {
            "total_iterations": state.get("iterations", 0),
            "task_results": [
                {
                    "id": r["id"],
                    "file": r["file"],
                    "action": r["action"],
                    "tool": r.get("tool"),
                    "arguments": r.get("arguments"),
                    "status": r["status"],
                    "attempts": r["attempts"],
                    "static_check": r.get("static_check"),
                    "diff_excerpt": r.get("diff", "")[:600],
                }
                for r in results
            ],
            "tool_trace": tool_trace[:25],
        },
        indent=2,
    )[:6000]

    agentic_scores = {}
    if DEEPEVAL_AVAILABLE:
        metrics = _build_agentic_deepeval_metrics()
        test_case = LLMTestCase(input=input_payload, actual_output=actual_payload)
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                raw_score = round(metric.score, 2) if metric.score is not None else fallbacks[name][0]
                reason = metric.reason or fallbacks[name][1]
            except Exception as e:
                raw_score, fb_reason = fallbacks[name]
                reason = f"{fb_reason} (DeepEval fallback due to: {e})"
            meta = classify_metric_score(name, raw_score)
            meta["reason"] = reason
            agentic_scores[name] = meta
    else:
        for name, (fb_score, fb_reason) in fallbacks.items():
            meta = classify_metric_score(name, fb_score)
            meta["reason"] = fb_reason
            agentic_scores[name] = meta

    return agentic_scores


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
    return round(max(0.25, 1.0 - 0.25 * (attempts - 1)), 2)


def render_evaluation_markdown(overall: dict, agentic_metrics: dict, report: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"**Custom score avg:** {overall['custom_score_avg']} / 1.0",
    ]
    if overall.get("deepeval_avg_score") is not None:
        lines.append(f"**DeepEval score avg:** {overall['deepeval_avg_score']} / 1.0")
    if overall.get("agentic_deepeval_avg") is not None:
        lines.append(f"**Agentic DeepEval avg:** {overall['agentic_deepeval_avg']} / 1.0")
    lines.append(f"**Tasks passed:** {overall['tasks_passed']}")
    lines.append(f"**Total time taken:** {overall['total_time_sec']} s")
    lines.append(f"**Total tasks:** {overall['total_tasks']}")
    lines.append(f"**Time per task:** {overall['time_per_task_sec']} s/task")
    lines.append("")
    lines.append("## Agentic Workflow Evaluation Matrix (DeepEval)")
    lines.append("")
    lines.append("| Metric | Score | Threshold | Bad Range | Good Range | Status | Band | Reason |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
    for name, data in agentic_metrics.items():
        sc = data["score"] if data["score"] is not None else "n/a"
        status = "PASS" if data["passed"] else "FAIL"
        reason = str(data.get("reason", "")).replace("\n", " ")[:160]
        lines.append(
            f"| **{name}** | {sc} | {data['threshold']:.2f} | {data['bad_range']} | "
            f"{data['good_range']} | {status} | {data['band']} | {reason} |"
        )
    lines.append("")
    for filename, data in report.items():
        lines.append(f"## {filename}")
        lines.append(f"- Tasks: {data['tasks']} | Attempts: {data['attempts']} | Custom score: {data['custom_score']}")
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
    W, H, PAD = 920, 300, 48
    n = len(records)
    if n == 0:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="120">'
                f'<text x="20" y="60" font-family="sans-serif" font-size="13">'
                f'No runs tracked yet.</text></svg>')

    def x_for(i):
        return PAD + (i / max(n - 1, 1)) * (W - 2 * PAD)

    def y_for(s):
        return H - PAD - s * (H - 2 * PAD)

    series_defs = [
        ("deepeval_avg_score", "DeepEval Avg", "#2563eb"),
        ("custom_score_avg", "Custom Avg", "#16a34a"),
        ("plan_quality", "Plan Quality", "#9333ea"),
        ("plan_adherence", "Plan Adherence", "#db2777"),
        ("tool_correctness", "Tool Correctness", "#ea580c"),
        ("argument_correctness", "Arg Correctness", "#0891b2"),
        ("step_efficiency", "Step Efficiency", "#4f46e5"),
        ("task_completion", "Task Completion", "#059669"),
    ]

    polylines = []
    legend_items = []
    for idx, (key, label, color) in enumerate(series_defs):
        pts = [(i, r.get(key)) for i, r in enumerate(records) if r.get(key) is not None]
        avg_str = ""
        if pts:
            poly = " ".join(f"{x_for(i):.1f},{y_for(s):.1f}" for i, s in pts)
            dots = "".join(
                f'<circle cx="{x_for(i):.1f}" cy="{y_for(s):.1f}" r="3" fill="{color}">'
                f'<title>Run {i + 1} {label}: {s:.2f}</title></circle>'
                for i, s in pts
            )
            polylines.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>{dots}')
            avg_val = sum(s for _, s in pts) / len(pts)
            avg_str = f" ({avg_val:.2f})"
        lx = PAD + (idx % 4) * 210
        ly = 14 + (idx // 4) * 16
        legend_items.append(
            f'<circle cx="{lx}" cy="{ly}" r="4" fill="{color}"/>'
            f'<text x="{lx + 8}" y="{ly + 4}" font-size="10.5" fill="#374151">{label}{avg_str}</text>'
        )

    gridlines = "".join(
        f'<line x1="{PAD}" y1="{y_for(g):.1f}" x2="{W - PAD}" y2="{y_for(g):.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        f'<text x="6" y="{y_for(g) + 4:.1f}" font-size="10" fill="#6b7280">{g:.2f}</text>'
        for g in (0.0, 0.25, 0.5, 0.70, 0.85, 1.0)
    )
    thresh_y = y_for(0.70)
    threshold_line = (
        f'<line x1="{PAD}" y1="{thresh_y:.1f}" x2="{W - PAD}" y2="{thresh_y:.1f}" '
        f'stroke="#dc2626" stroke-width="1.2" stroke-dasharray="4,4"/>'
        f'<text x="{W - PAD - 85}" y="{thresh_y - 4:.1f}" font-size="10" fill="#dc2626">Threshold 0.70</text>'
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
{gridlines}
{threshold_line}
{"".join(polylines)}
{"".join(legend_items)}
<text x="{PAD}" y="{H - 10}" font-size="11" fill="#374151">Run 1</text>
<text x="{W - PAD - 40}" y="{H - 10}" font-size="11" fill="#374151">Run {n}</text>
</svg>'''


def _format_metric_cell(metric_name: str, val: Optional[float]) -> str:
    if val is None:
        return "<td>n/a</td>"
    info = classify_metric_score(metric_name, val)
    if info["band"] == "Good":
        bg, fg = "#dcfce7", "#166534"
    elif info["passed"]:
        bg, fg = "#fef9c3", "#854d0e"
    else:
        bg, fg = "#fee2e2", "#991b1b"
    return (
        f'<td style="background:{bg};color:{fg};font-weight:600;" '
        f'title="{metric_name}: {info[\'band\']} (Threshold {info[\'threshold\']:.2f}, Good {info[\'good_range\']}, Bad {info[\'bad_range\']})">'
        f'{val:.2f} <span style="font-size:10px;font-weight:400;">({info["band"]})</span></td>'
    )


def render_benchmark_html(records: list) -> str:
    def avg_of(key: str):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else "n/a"

    all_time_deepeval = avg_of("deepeval_avg_score")
    all_time_custom = avg_of("custom_score_avg")
    all_time_tpt = avg_of("time_per_task_sec")
    all_time_pq = avg_of("plan_quality")
    all_time_pa = avg_of("plan_adherence")
    all_time_tc = avg_of("tool_correctness")
    all_time_ac = avg_of("argument_correctness")
    all_time_se = avg_of("step_efficiency")
    all_time_tcomp = avg_of("task_completion")

    spec_rows = []
    for name, spec in AGENTIC_METRIC_SPECS.items():
        avg_val = avg_of(spec["key"])
        spec_rows.append(
            f"<tr><td><b>{name}</b></td>"
            f"<td style='color:#991b1b;background:#fef2f2;'>{spec['bad_label']}</td>"
            f"<td style='color:#166534;background:#f0fdf4;'>{spec['good_label']}</td>"
            f"<td><b>{spec['threshold']:.2f}</b></td>"
            f"<td><b>{avg_val}</b></td></tr>"
        )

    rows = []
    for i, r in enumerate(records):
        cfg = r.get("llm_config", {})
        rows.append(
            f"<tr><td>{i + 1}</td><td>{r.get('timestamp', '')}</td>"
            f"<td>{cfg.get('model', '')}</td><td>{cfg.get('num_ctx', '')}</td>"
            f"<td>{str(r.get('request', r.get('app_brief', '')))[:55]}</td>"
            f"<td>{r.get('tasks_passed', r.get('files_passed', ''))}</td>"
            f"<td>{r.get('total_tasks', 'n/a')}</td>"
            f"<td>{r.get('total_time_sec', 'n/a')}</td>"
            f"<td>{r.get('time_per_task_sec', 'n/a')}</td>"
            f"<td>{r.get('custom_score_avg', 'n/a')}</td>"
            f"<td>{r.get('deepeval_avg_score', 'n/a')}</td>"
            f"{_format_metric_cell('Plan Quality', r.get('plan_quality'))}"
            f"{_format_metric_cell('Plan Adherence', r.get('plan_adherence'))}"
            f"{_format_metric_cell('Tool Correctness', r.get('tool_correctness'))}"
            f"{_format_metric_cell('Argument Correctness', r.get('argument_correctness'))}"
            f"{_format_metric_cell('Step Efficiency', r.get('step_efficiency'))}"
            f"{_format_metric_cell('Task Completion', r.get('task_completion'))}</tr>"
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Loop Workflow — Benchmark History</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111827; max-width: 1450px; }}
table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }}
th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 12.5px; }}
th {{ background: #f9fafb; }}
.matrix-table {{ width: auto; min-width: 640px; margin-bottom: 1.5rem; }}
.summary-bar {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px 14px; border-radius: 6px; font-size: 13px; line-height: 1.6; }}
</style></head>
<body>
<h1>Benchmark History &amp; DeepEval Agentic Matrix</h1>
<div class="summary-bar">
<b>{len(records)}</b> run(s) tracked —
All-time avg custom score: <b>{all_time_custom}</b> &nbsp;|&nbsp;
All-time avg DeepEval score: <b>{all_time_deepeval}</b> &nbsp;|&nbsp;
All-time avg time/task: <b>{all_time_tpt} s</b><br>
Agentic Metrics All-Time Avgs —
Plan Quality: <b>{all_time_pq}</b> &nbsp;|&nbsp;
Plan Adherence: <b>{all_time_pa}</b> &nbsp;|&nbsp;
Tool Correctness: <b>{all_time_tc}</b> &nbsp;|&nbsp;
Argument Correctness: <b>{all_time_ac}</b> &nbsp;|&nbsp;
Step Efficiency: <b>{all_time_se}</b> &nbsp;|&nbsp;
Task Completion: <b>{all_time_tcomp}</b>
</div>

<h3>DeepEval Agentic Evaluation Reference Matrix</h3>
<table class="matrix-table">
<tr><th>Metric</th><th>Bad Range</th><th>Good Range</th><th>Threshold</th><th>All-Time Avg</th></tr>
{"".join(spec_rows)}
</table>

{render_benchmark_svg(records)}
<table>
<tr><th>#</th><th>Timestamp</th><th>Model</th><th>num_ctx</th><th>Request</th>
<th>Tasks passed</th><th>Total tasks</th><th>Total time (s)</th><th>Time/task (s)</th>
<th>Custom avg</th><th>DeepEval avg</th>
<th>Plan Quality</th><th>Plan Adherence</th><th>Tool Correctness</th>
<th>Argument Correctness</th><th>Step Efficiency</th><th>Task Completion</th></tr>
{"".join(rows)}
</table>
</body></html>"""


def evaluator_node(state: AgentState):
    total_time = round(time.perf_counter() - state["start_time"], 2)
    print("\n📊 [Evaluator] Scoring the finished work and Agentic DeepEval Matrix...")
    run_dir = state["run_dir"]
    root = Path(state["project_address"]).resolve()
    manifest = load_manifest(state)
    results = state.get("task_results", [])
    planned_tasks = state.get("planned_tasks", [])
    tool_trace = state.get("tool_trace", [])
    total_tasks = state.get("total_tasks", len(results))
    tasks_passed = sum(1 for r in results if r["status"] == "passed")

    plan_text = ""
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        if not planned_tasks:
            planned_tasks = plan_data.get("execution_order", [])
        plan_text = json.dumps(planned_tasks, indent=2)
    context = (
        f"USER REQUEST:\n{state['user_request']}\n\n"
        f"MANIFEST:\n{manifest_brief(manifest)}\n\n"
        f"PLAN:\n{plan_text[:3000]}"
    )
    file_metrics = _build_file_deepeval_metrics() if DEEPEVAL_AVAILABLE else []

    file_tasks: dict = {}
    for r in results:
        if r["status"] == "passed":
            file_tasks.setdefault(r["file"], []).append(r)

    report = {}
    for filename, rs in file_tasks.items():
        try:
            code, _ = read_text_lf(resolve_in_project(root, filename))
        except (OSError, UnicodeDecodeError, ValueError):
            code = ""
        scores = [compute_custom_score(r["attempts"]) for r in rs]
        report[filename] = {
            "tasks": len(rs),
            "attempts": sum(r["attempts"] for r in rs),
            "custom_score": round(sum(scores) / len(scores), 2),
            "deepeval": deepeval_score_file(filename, code[:6000], context, file_metrics),
        }

    agentic_metrics = evaluate_agentic_workflow(state, manifest, planned_tasks, results, tool_trace)
    agentic_score_vals = [m["score"] for m in agentic_metrics.values() if m.get("score") is not None]
    agentic_avg = round(sum(agentic_score_vals) / len(agentic_score_vals), 2) if agentic_score_vals else None

    custom_scores = [f["custom_score"] for f in report.values()]
    overall = {
        "tasks_passed": f"{tasks_passed}/{total_tasks}",
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "custom_score_avg": round(sum(custom_scores) / len(custom_scores), 2) if custom_scores else 0.0,
        "total_tasks": total_tasks,
        "total_time_sec": total_time,
        "time_per_task_sec": round(total_time / total_tasks, 2) if total_tasks else 0.0,
        "agentic_deepeval_avg": agentic_avg,
    }
    for name, spec in AGENTIC_METRIC_SPECS.items():
        overall[spec["key"]] = agentic_metrics[name]["score"]

    file_deval_scores = [
        m["score"] for f in report.values() if f.get("deepeval")
        for m in f["deepeval"].values() if m and m.get("score") is not None
    ]
    combined_deval = file_deval_scores + agentic_score_vals
    if combined_deval:
        overall["deepeval_avg_score"] = round(sum(combined_deval) / len(combined_deval), 2)

    eval_payload = {
        "overall": overall,
        "agentic_metrics": agentic_metrics,
        "files": report,
        "tasks": results,
        "tool_trace": tool_trace,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    (run_dir / "evaluation.md").write_text(
        render_evaluation_markdown(overall, agentic_metrics, report), encoding="utf-8"
    )
    write_log(run_dir, "Evaluator", json.dumps({"overall": overall, "agentic_metrics": agentic_metrics}, indent=2))

    benchmark_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "request": state.get("user_request", "")[:200],
        "llm_config": {
            "model": llm.model,
            "num_ctx": llm.num_ctx,
            "num_predict": llm.num_predict,
            "temperature": llm.temperature,
        },
        "tasks_passed": overall["tasks_passed"],
        "total_tasks": total_tasks,
        "total_time_sec": total_time,
        "time_per_task_sec": overall["time_per_task_sec"],
        "custom_score_avg": overall["custom_score_avg"],
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "deepeval_avg_score": overall.get("deepeval_avg_score"),
        "agentic_deepeval_avg": agentic_avg,
        "plan_quality": overall.get("plan_quality"),
        "plan_adherence": overall.get("plan_adherence"),
        "tool_correctness": overall.get("tool_correctness"),
        "argument_correctness": overall.get("argument_correctness"),
        "step_efficiency": overall.get("step_efficiency"),
        "task_completion": overall.get("task_completion"),
        "agentic_metrics": agentic_metrics,
        "per_file": {fn: {"tasks": d.get("tasks"), "attempts": d.get("attempts")} for fn, d in report.items()},
    }
    history = append_benchmark_record(benchmark_record)
    html_report = render_benchmark_html(history)
    (benchmark_history_path().parent / "benchmark_history.html").write_text(html_report, encoding="utf-8")
    (run_dir / "evaluation.html").write_text(html_report, encoding="utf-8")

    print(f"   Tasks passed: {overall['tasks_passed']}")
    print(f"   Total time: {total_time}s | Total tasks: {total_tasks} | Time/task: {overall['time_per_task_sec']}s")
    print(f"   Custom score avg: {overall['custom_score_avg']}/1.0")
    if "deepeval_avg_score" in overall:
        print(f"   DeepEval score avg: {overall['deepeval_avg_score']}/1.0")
    for name, data in agentic_metrics.items():
        print(
            f"   • {name:20s}: {data['score']:.2f} "
            f"[{data['band']} | threshold {data['threshold']:.2f} | good {data['good_range']}]"
        )
    print(f"   📈 Benchmark: {len(history)} run(s) tracked — see benchmark_history.jsonl / benchmark_history.html")
    return {}


# ==========================================
# 5. CONDITIONAL ROUTER & GRAPH
# ==========================================
def route_after_review(state: AgentState) -> str:
    if state.get("issue_report", "").strip():
        return "retry"
    if state.get("plan_stack"):
        return "next"
    return "done"


workflow = StateGraph(AgentState)
workflow.add_node("requirement", requirement_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("editor", editor_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("evaluator", evaluator_node)

workflow.set_entry_point("requirement")
workflow.add_edge("requirement", "analyzer")
workflow.add_edge("analyzer", "planner")
workflow.add_edge("planner", "editor")
workflow.add_edge("editor", "reviewer")
workflow.add_conditional_edges(
    "reviewer", route_after_review, {"retry": "editor", "next": "editor", "done": "evaluator"}
)
workflow.add_edge("evaluator", END)

app = workflow.compile()


# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic multi-agent codebase editor (LangGraph + Ollama).")
    parser.add_argument("--project", help="Path to the target local repository")
    parser.add_argument("--request", help="The change to implement")
    args = parser.parse_args()

    project_address = args.project or input("\nPath to the target repository:\n> ").strip()
    user_request = args.request or input("\nWhat change do you want to make?\n> ").strip()

    base_dir = Path.cwd() / "SDLC_Runs"
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    write_log(run_dir, "SYSTEM", f"Starting edit run on '{project_address}' for: '{user_request}'")

    print(f"\n🚀 Starting run in: {run_dir}")
    app.invoke(
        {"project_address": project_address, "user_request": user_request, "run_dir": run_dir},
        config={"recursion_limit": 500},
    )
    print(f"\n🎉 Run Complete! Check execution_log.txt in {run_dir}")
    print(f"🗄️  Backups: {run_dir / 'backups'}")
    print(f"📊 Evaluation report: {run_dir / 'evaluation.md'}")
    print(f"📈 Benchmark history: {Path(__file__).resolve().parent / 'benchmark_history.html'}")