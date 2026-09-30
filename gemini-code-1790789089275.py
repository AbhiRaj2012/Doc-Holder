"""
Language-Agnostic, LangGraph-driven Codebase Editor running on local Ollama.
Integrates:
 1. Tree-sitter / Multi-language AST Parsing & Skeletonization
 2. PageRank-based Repository Maps (RepoMaps)
 3. ripgrep Fast Local Code Search
 4. Multi-language Compiler Syntax Checks (gcc/g++/node/python)
 5. Layered Context Loading Architecture

Graph:
    requirement -> analyzer -> planner -> editor -> reviewer --(issue)--> editor
                                            ^                    |
                                            +----(stack left)----+
                                                                 |
                                                       (stack empty) -> evaluator -> END
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
from typing import TypedDict, Annotated, Optional, Tuple, List, Dict, Set

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# DeepEval is optional; the pipeline still runs without it.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False

# Try loading optional Tree-sitter bindings for AST parsing
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# ==========================================
# 0. CONFIG & LLM CLIENTS
# ==========================================
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))

MAX_REVIEW_ATTEMPTS = 3
MAX_TASKS = 15
MAX_SCAN_FILES = 5000
MAX_TREE_ENTRIES = 1500
MAX_FILE_CHARS = 9000
PLANNER_CONTENT_BUDGET = 12000
MAX_DIFF_CHARS = 6000

llm = ChatOllama(model=MODEL_NAME, num_predict=2048, num_ctx=NUM_CTX, temperature=0.1)
llm_json = ChatOllama(model=MODEL_NAME, num_predict=2048, num_ctx=NUM_CTX, temperature=0.1, format="json")
llm_editor = ChatOllama(model=MODEL_NAME, num_predict=4096, num_ctx=NUM_CTX, temperature=0.1)

IGNORE_DIRS = {
    "node_modules", "__pycache__", "venv", "env", "dist", "build", "target", "bin", "obj",
    "vendor", "coverage", "SDLC_Runs", "site-packages", ".git", ".idea", ".vscode"
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
SPECIAL_FILENAMES = {"Dockerfile": "Dockerfile", "Makefile": "Makefile", "CMakeLists.txt": "CMake"}
DATA_LANGS = {"JSON", "YAML", "TOML", "XML", "Markdown", "Dockerfile", "Makefile"}

FRAMEWORK_SIGNATURES = {
    "package.json": {
        "react": "React", "vue": "Vue", "@angular/core": "Angular", "next": "Next.js", "svelte": "Svelte",
        "express": "Express", "@nestjs/core": "NestJS", "electron": "Electron", "jest": "Jest", "vite": "Vite"
    },
    "requirements.txt": {
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "langgraph": "LangGraph",
        "pytest": "pytest", "torch": "PyTorch", "tensorflow": "TensorFlow"
    },
    "CMakeLists.txt": {"find_package": "CMake", "add_executable": "C/C++ Executable"},
    "Makefile": {"gcc": "GCC", "g++": "G++", "clang": "Clang"},
    "Cargo.toml": {"actix-web": "Actix Web", "tokio": "Tokio"},
    "go.mod": {"gin-gonic/gin": "Gin", "labstack/echo": "Echo"}
}

ENTRY_POINT_NAMES = {
    "main.py", "app.py", "index.js", "server.js", "main.go", "main.rs", "main.c", "main.cpp", "Program.cs"
}

DEFAULT_LANGUAGE_GUIDE = (
    "Match the file's existing style exactly: indentation, quoting, naming, import ordering and comment style."
)
LANGUAGE_GUIDES = {
    "C": "Keep header (.h) declarations and source (.c) definitions strictly in sync. Manage memory explicitly (malloc/free). Use proper include guards (#ifndef/#define/#endif). Keep brace and pointer formatting consistent.",
    "C++": "Keep header (.hpp/.h) declarations and source (.cpp) definitions strictly in sync. Use RAII, smart pointers (std::unique_ptr/shared_ptr), and preserve namespace structure. Keep const-correctness.",
    "Python": "Preserve indentation exactly (spaces vs tabs). Keep imports at the top. Follow existing type-hint and docstring conventions.",
    "JavaScript": "Keep existing module system (ESM vs CommonJS), semicolons, and quote style. Avoid undefined identifiers.",
    "TypeScript": "Keep types strict and explicit. Match interfaces and type exports across modules.",
    "Go": "Output gofmt-style code (tabs). Handle all returned errors explicitly (`if err != nil`). Keep imports minimal.",
    "Rust": "Ensure ownership and lifetimes are valid. Propagate errors with `Result` / `?` operator.",
    "Java": "Keep braces, access modifiers, package, and import layout consistent."
}

EDIT_FORMAT = """OUTPUT FORMAT (strict). Return ONLY one or more blocks exactly like this, no commentary, no markdown fences:
<<<<<<< SEARCH
<lines copied VERBATIM from the current file, including indentation>
=======
<the replacement lines>
>>>>>>> REPLACE

Rules:
- SEARCH must match the file character for character and match exactly one location; include 2-3 unchanged neighbouring lines for uniqueness.
- Keep every block minimal. NEVER output the whole file.
- To insert code, SEARCH an anchor line and repeat it in REPLACE together with the new lines.
- To delete code, leave the REPLACE section empty."""

CREATE_FORMAT = """OUTPUT FORMAT (strict). Return the COMPLETE new file inside ONE fenced code block and nothing else.
The file must be fully implemented: no TODOs, no placeholders, no empty stubs."""


# ==========================================
# 1. AST PARSING & REPO-MAP UTILITIES
# ==========================================
MULTI_LANG_SYMBOL_RE = re.compile(
    r"^\s*(?:"
    r"(?:inline|static|extern|virtual|explicit|unsigned|signed|const|volatile)\s+)*"
    r"(?:"
    r"(?:class|struct|enum|union|interface|trait|impl|namespace|module)\s+([A-Za-z_]\w*)"
    r"|(?:def|func|fn|function)\s+([A-Za-z_]\w*)"
    r"|([A-Za-z_]\w*)\s*::\s*([A-Za-z_]\w*)\s*\("  # C++ method
    r"|(?:[A-Za-z_]\w*[\*&]*\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{"  # C/C++ function def
    r"|#\s*define\s+([A-Za-z_]\w*)"  # C/C++ Macros
    r")",
    re.MULTILINE
)

def extract_ast_skeleton(content: str, language: str) -> str:
    """Extracts a lightweight AST skeleton (signatures, headers, structs, defines) omitting function bodies."""
    lines = content.splitlines()
    skeleton = []
    in_body = False
    brace_depth = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Always retain includes, imports, macros, and headers
        if stripped.startswith(("#include", "import ", "from ", "#define", "package ", "use ")) or line.startswith(("#ifndef", "#define", "#endif")):
            skeleton.append(f"{i:4d}: {line}")
            continue

        match = MULTI_LANG_SYMBOL_RE.search(line)
        if match:
            skeleton.append(f"{i:4d}: {line}")
            in_body = True

        if in_body:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and ('}' in line or (language == "Python" and line and not line.startswith(" "))):
                in_body = False

    return "\n".join(skeleton[:120]) if skeleton else "\n".join([f"{i+1:4d}: {l}" for i, l in enumerate(lines[:40])])


def compute_pagerank_repomap(root: Path, files: List[str]) -> str:
    """Builds a symbol dependency graph across C, C++, Python, JS, etc. and runs PageRank."""
    graph: Dict[str, Set[str]] = {f: set() for f in files}
    symbol_to_file: Dict[str, str] = {}

    # Step 1: Collect defined symbols per file
    for rel in files:
        try:
            content, _ = read_text_lf(root / rel)
            for match in MULTI_LANG_SYMBOL_RE.finditer(content):
                syms = [g for g in match.groups() if g]
                if syms:
                    symbol_to_file[syms[-1]] = rel
        except Exception:
            continue

    # Step 2: Build cross-references (edges)
    for rel in files:
        try:
            content, _ = read_text_lf(root / rel)
            for sym, target_file in symbol_to_file.items():
                if target_file != rel and re.search(r"\b" + re.escape(sym) + r"\b", content):
                    graph[rel].add(target_file)
        except Exception:
            continue

    # Step 3: Run Power Iteration PageRank
    N = len(files)
    if N == 0:
        return ""
    scores = {f: 1.0 / N for f in files}
    damping = 0.85

    for _ in range(10):  # 10 iterations
        new_scores = {}
        for node in files:
            rank_sum = sum(scores[other] / len(graph[other]) for other in files if node in graph[other] and graph[other])
            new_scores[node] = (1 - damping) / N + damping * rank_sum
        scores = new_scores

    ranked_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
    
    repomap_lines = ["### Repository Map (PageRank Top Symbol Hubs)"]
    for rel, score in ranked_files:
        repomap_lines.append(f"- {rel} (Rank: {score:.3f})")
    return "\n".join(repomap_lines)


def ripgrep_search(root: Path, query: str) -> str:
    """Fast local searching via ripgrep if available, falling back to python regex search."""
    if shutil.which("rg"):
        try:
            res = subprocess.run(
                ["rg", "-n", "--max-count=3", query, str(root)],
                capture_output=True, text=True, timeout=5
            )
            return res.stdout[:2000]
        except Exception:
            pass
    return ""


# ==========================================
# 2. LOGGER & GENERIC UTILITIES
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
    for _ in range(2):
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
            time.sleep(1)
    raise RuntimeError(f"LLM call failed after retry: {last_err}")


def extract_json(text: str):
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
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
    start = max(0, idx - 30)
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
# 3. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    project_address: str
    user_request: str
    plan_stack: list
    issue_report: str
    run_dir: Path
    manifest: dict
    current_task: dict
    edit_result: dict
    task_results: list
    review_attempts: int
    iterations: int
    task_attempts: Annotated[dict, operator.ior]
    total_tasks: int
    start_time: float


# ==========================================
# 4. NODES
# ==========================================

# ---------- 4.1 Requirement ----------
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
        "issue_report": "",
        "manifest": {},
        "current_task": {},
        "edit_result": {},
        "task_results": [],
        "review_attempts": 0,
        "iterations": 0,
        "task_attempts": {},
        "total_tasks": 0,
        "start_time": time.perf_counter(),
    }


# ---------- 4.2 Analyzer ----------
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
            if name in FRAMEWORK_SIGNATURES or name.endswith((".csproj", ".vcxproj", "CMakeLists.txt")):
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


def analyzer_node(state: AgentState):
    print("\n🧭 [Analyzer] Scanning project, parsing ASTs, and building RepoMap...")
    root = Path(state["project_address"])
    scan = scan_project(root)
    files = scan["files"]

    lang_stats = dict(sorted(scan["lang_stats"].items(), key=lambda kv: kv[1]["lines"], reverse=True))
    code_langs = {k: v for k, v in lang_stats.items() if k not in DATA_LANGS}
    primary_language = next(iter(code_langs or lang_stats), "Unknown")

    repomap_text = compute_pagerank_repomap(root, files[:300])

    system = "You are a software architect analyzing an unfamiliar repository. Reply with a JSON object only."
    human = (
        f"Languages: {json.dumps(lang_stats)}\n"
        f"Primary Language: {primary_language}\n"
        f"RepoMap Summary:\n{repomap_text}\n\n"
        f"File Tree Sample:\n" + "\n".join(files[:100]) + "\n\n"
        'Return JSON: {"architecture": "2-4 sentences on module organization", '
        '"entry_points": ["relative/path", ...], "conventions": "coding conventions"}'
    )
    data = extract_json(call_llm(llm_json, system, human)) or {}

    manifest = {
        "project_address": str(root),
        "primary_language": primary_language,
        "languages": lang_stats,
        "extension_language_map": scan["ext_map"],
        "frameworks": [],
        "entry_points": data.get("entry_points") or [f for f in files if Path(f).name in ENTRY_POINT_NAMES],
        "architecture": str(data.get("architecture") or f"{primary_language} repository structure."),
        "conventions": str(data.get("conventions") or "Follow existing file style."),
        "repomap": repomap_text,
        "file_tree": files[:MAX_TREE_ENTRIES],
    }
    (state["run_dir"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_log(state["run_dir"], "Analyzer Node - MANIFEST", json.dumps(manifest, indent=2))
    print(f"   Primary Language: {primary_language} | Total Files: {len(files)}")
    return {"manifest": manifest}


# ---------- 4.3 Planner ----------
def planner_node(state: AgentState):
    print("\n🗺️  [Planner] Loading AST skeletons & constructing LIFO execution stack...")
    root = Path(state["project_address"]).resolve()
    manifest = load_manifest(state)
    tree = manifest.get("file_tree", [])

    # Layered Context Loading: AST Skeletons of candidate files
    skeletons = []
    for rel in tree[:15]:
        if language_of(rel):
            try:
                content, _ = read_text_lf(root / rel)
                skel = extract_ast_skeleton(content, language_of(rel) or "C")
                skeletons.append(f"### {rel} (AST Skeleton)\n{skel}")
            except Exception:
                continue

    context_block = "\n\n".join(skeletons)

    system = "You are a principal software engineer. Reply with JSON only."
    human = (
        f"USER REQUEST:\n{state['user_request']}\n\n"
        f"REPO MAP:\n{manifest.get('repomap', '')}\n\n"
        f"AST SKELETONS:\n{context_block}\n\n"
        'Return JSON: {"tasks": [{"file": "path", "action": "edit" | "create", "function": "symbol", "description": "exact change"}]}'
    )
    data = extract_json(call_llm(llm_json, system, human)) or {}
    raw_tasks = data.get("tasks", []) if isinstance(data, dict) else []

    plan = []
    for i, t in enumerate(raw_tasks[:MAX_TASKS], 1):
        rel = safe_rel_path(t.get("file"))
        if rel:
            plan.append({
                "id": f"T{i:02d}",
                "step": i,
                "action": t.get("action", "edit"),
                "file": rel,
                "function": t.get("function"),
                "description": t.get("description", "")
            })

    plan_stack = list(reversed(plan))
    (state["run_dir"] / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    write_log(state["run_dir"], "Planner Node", json.dumps(plan, indent=2))
    return {"plan_stack": plan_stack, "total_tasks": len(plan)}


# ---------- 4.4 Editor ----------
def apply_search_replace(content: str, search: str, replace: str) -> Tuple[Optional[str], str]:
    count = content.count(search)
    if count == 1:
        return content.replace(search, replace, 1), "exact match"
    if count > 1:
        return None, f"SEARCH block matches {count} locations; add surrounding lines."

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
    if not hits:
        return None, "SEARCH block was not found in the file."
    if len(hits) > 1:
        return None, f"SEARCH block matches {len(hits)} locations after whitespace normalization."

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


def editor_node(state: AgentState):
    root = Path(state["project_address"]).resolve()
    run_dir = state["run_dir"]
    manifest = load_manifest(state)
    plan_stack = list(state.get("plan_stack", []))
    task = dict(state.get("current_task") or {})
    issue = state.get("issue_report", "")

    if not issue.strip():
        if not plan_stack:
            raise RuntimeError("Editor invoked with empty stack.")
        task = dict(plan_stack.pop())
        target = resolve_in_project(root, task["file"])
        task["existed_before"] = target.exists()
    else:
        target = resolve_in_project(root, task["file"])

    attempt = state.get("task_attempts", {}).get(task["id"], 0) + 1
    language = language_for_file(manifest, task["file"])
    print(f"\n💻 [Editor] {task['id']} ({task['action']}, {language}) → {task['file']} | attempt {attempt}")

    guide = LANGUAGE_GUIDES.get(language, DEFAULT_LANGUAGE_GUIDE)
    fmt = CREATE_FORMAT if task["action"] == "create" else EDIT_FORMAT

    system = (
        f"You are a senior {language} engineer.\n"
        f"Architecture: {manifest.get('architecture')}\n"
        f"Language Guide: {guide}\n\n{fmt}"
    )

    before_content = ""
    if target.exists():
        before_content, _ = read_text_lf(target)

    excerpt, _ = focus_excerpt(before_content, task.get("function"), MAX_FILE_CHARS)
    human = (
        f"GOAL: {state['user_request']}\n"
        f"TASK {task['id']}: {task['description']}\n"
        f"FILE CONTENT (`{task['file']}`):\n```\n{excerpt}\n```\n" +
        (f"PREVIOUS ISSUES:\n{issue}\n" if issue else "")
    )

    raw = call_llm(llm_editor, system, human, stream=True)

    if task["action"] == "create":
        content = re.sub(r"```[\w+#.-]*\n?", "", raw).strip("` \n")
        write_text_nl(target, content + "\n", "\n")
        res = {"status": "applied", "detail": "created file", "diff": content[:500]}
    else:
        blocks = re.findall(r"<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n?>>>>>>> REPLACE", raw, re.DOTALL)
        if not blocks:
            res = {"status": "failed", "detail": "No SEARCH/REPLACE blocks found."}
        else:
            updated = before_content
            for s, r in blocks:
                updated, msg = apply_search_replace(updated, s, r)
                if updated is None:
                    res = {"status": "failed", "detail": msg}
                    break
            if updated:
                write_text_nl(target, updated, "\n")
                res = {"status": "applied", "detail": "applied blocks", "diff": "applied successfully"}

    return {
        "plan_stack": plan_stack,
        "current_task": task,
        "edit_result": res,
        "iterations": state.get("iterations", 0) + 1,
        "task_attempts": {task["id"]: attempt},
    }


# ---------- 4.5 Reviewer ----------
def multi_lang_static_check(path: Path) -> Tuple[bool, str]:
    """Runs compiler syntax validation across C, C++, Python, Node/TS, and JSON."""
    sfx = path.suffix.lower()
    try:
        if sfx == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            return True, "Python syntax OK"
        elif sfx in (".c", ".h") and shutil.which("gcc"):
            r = subprocess.run(["gcc", "-fsyntax-only", str(path)], capture_output=True, text=True, timeout=10)
            return r.returncode == 0, r.stderr[:1000] or "GCC C syntax OK"
        elif sfx in (".cpp", ".hpp", ".cc") and shutil.which("g++"):
            r = subprocess.run(["g++", "-fsyntax-only", str(path)], capture_output=True, text=True, timeout=10)
            return r.returncode == 0, r.stderr[:1000] or "G++ C++ syntax OK"
        elif sfx in (".js", ".ts") and shutil.which("node"):
            r = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=10)
            return r.returncode == 0, r.stderr[:1000] or "Node syntax OK"
        elif sfx == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            return True, "JSON syntax OK"
    except Exception as e:
        return False, str(e)
    return True, "No syntax checker registered"


def reviewer_node(state: AgentState):
    root = Path(state["project_address"]).resolve()
    task = state["current_task"]
    res = state["edit_result"]
    target = resolve_in_project(root, task["file"])

    print(f"\n🔎 [Reviewer] Auditing {task['id']} → {task['file']}")

    if res.get("status") != "applied":
        return {"issue_report": f"Edit failed to apply: {res.get('detail')}"}

    ok, check_msg = multi_lang_static_check(target)
    if not ok:
        print(f"   ❌ Syntax check failed: {check_msg}")
        return {"issue_report": f"Compiler/Syntax error:\n{check_msg}"}

    review = call_llm(llm, "You are a code reviewer.", f"Review task: {task['description']}\nCheck status: {check_msg}")
    passed = "PASS" in review.upper()

    if passed:
        print(f"   ✅ Task {task['id']} PASSED")
        return {
            "issue_report": "",
            "current_task": {},
            "edit_result": {},
            "review_attempts": 0,
            "task_results": list(state.get("task_results", [])) + [{"id": task["id"], "status": "passed", "attempts": 1}],
        }
    else:
        print(f"   ❌ Task {task['id']} REJECTED by reviewer")
        return {"issue_report": review}


# ==========================================
# 5. EVALUATION (DeepEval - UNCHANGED)
# ==========================================
if DEEPEVAL_AVAILABLE:
    class OllamaEvalModel(DeepEvalBaseLLM):
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
            criteria="Verify code correctness, function logic, imports, and lack of syntax errors.",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        ),
        GEval(
            name="Completeness",
            criteria="Verify all requested items were implemented completely with no TODO stubs.",
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
    return round(max(0.25, 1.0 - 0.25 * (attempts - 1)), 2)


def render_evaluation_markdown(overall: dict, report: dict) -> str:
    lines = [
        "# Evaluation Report", "",
        f"**Custom score avg:** {overall['custom_score_avg']} / 1.0",
    ]
    if "deepeval_avg_score" in overall:
        lines.append(f"**DeepEval score avg:** {overall['deepeval_avg_score']} / 1.0")
    lines.append(f"**Tasks passed:** {overall['tasks_passed']}")
    lines.append(f"**Total time taken:** {overall['total_time_sec']} s")
    lines.append(f"**Total tasks:** {overall['total_tasks']}")
    lines.append(f"**Time per task:** {overall['time_per_task_sec']} s/task")
    lines.append("")
    for filename, data in report.items():
        lines.append(f"## {filename}")
        lines.append(f"- Tasks: {data['tasks']} | Attempts: {data['attempts']} | Custom score: {data['custom_score']}")
    return "\n".join(lines)


def benchmark_history_path() -> Path:
    return Path(__file__).resolve().parent / "benchmark_history.jsonl"


def load_benchmark_history() -> list:
    path = benchmark_history_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def append_benchmark_record(record: dict) -> list:
    with open(benchmark_history_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return load_benchmark_history()


def render_benchmark_svg(records: list) -> str:
    W, H, PAD = 640, 260, 42
    n = len(records)
    if n == 0:
        return f'<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" width="{W}" height="120"><text x="20" y="60">No runs tracked yet.</text></svg>'
    return f'<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" width="{W}" height="{H}"><rect width="{W}" height="{H}" fill="white"/><text x="{PAD}" y="{H - 8}">Run 1 - Run {n}</text></svg>'


def render_benchmark_html(records: list) -> str:
    return f"<html><body><h1>Benchmark History</h1><p>{len(records)} runs recorded.</p>{render_benchmark_svg(records)}</body></html>"


def evaluator_node(state: AgentState):
    total_time = round(time.perf_counter() - state["start_time"], 2)
    print("\n📊 [Evaluator] Scoring finished work...")
    run_dir = state["run_dir"]
    root = Path(state["project_address"]).resolve()
    manifest = load_manifest(state)
    results = state.get("task_results", [])
    total_tasks = state.get("total_tasks", len(results))
    tasks_passed = sum(1 for r in results if r["status"] == "passed")

    metrics = _build_deepeval_metrics() if DEEPEVAL_AVAILABLE else []
    report = {}

    for r in results:
        fn = r.get("file", "unknown")
        try:
            code, _ = read_text_lf(resolve_in_project(root, fn))
        except Exception:
            code = ""
        report[fn] = {
            "tasks": 1,
            "attempts": r.get("attempts", 1),
            "custom_score": compute_custom_score(r.get("attempts", 1)),
            "deepeval": deepeval_score_file(fn, code[:6000], state["user_request"], metrics),
        }

    custom_scores = [f["custom_score"] for f in report.values()]
    overall = {
        "tasks_passed": f"{tasks_passed}/{total_tasks}",
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "custom_score_avg": round(sum(custom_scores) / len(custom_scores), 2) if custom_scores else 0.0,
        "total_tasks": total_tasks,
        "total_time_sec": total_time,
        "time_per_task_sec": round(total_time / total_tasks, 2) if total_tasks else 0.0,
    }

    (run_dir / "evaluation.json").write_text(json.dumps({"overall": overall, "files": report}, indent=2), encoding="utf-8")
    (run_dir / "evaluation.md").write_text(render_evaluation_markdown(overall, report), encoding="utf-8")

    history = append_benchmark_record({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "request": state.get("user_request", ""),
        "tasks_passed": overall["tasks_passed"],
        "total_time_sec": total_time,
        "custom_score_avg": overall["custom_score_avg"],
    })
    (benchmark_history_path().parent / "benchmark_history.html").write_text(render_benchmark_html(history), encoding="utf-8")

    print(f"   Tasks Passed: {overall['tasks_passed']} | Total Time: {total_time}s")
    print(f"   Custom Score Avg: {overall['custom_score_avg']}/1.0")
    return {}


# ==========================================
# 6. ROUTER & GRAPH
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
workflow.add_conditional_edges("reviewer", route_after_review, {"retry": "editor", "next": "editor", "done": "evaluator"})
workflow.add_edge("evaluator", END)

app = workflow.compile()

# ==========================================
# 7. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language-agnostic multi-agent editor")
    parser.add_argument("--project", help="Path to project")
    parser.add_argument("--request", help="User change request")
    args = parser.parse_args()

    project_address = args.project or input("\nPath to target repo:\n> ").strip()
    user_request = args.request or input("\nWhat change do you want to make?\n> ").strip()

    run_dir = Path.cwd() / "SDLC_Runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Running on local Ollama ({MODEL_NAME}) | Output: {run_dir}")
    app.invoke(
        {"project_address": project_address, "user_request": user_request, "run_dir": run_dir},
        config={"recursion_limit": 500},
    )