import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
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


# ==========================================
# 0. CONFIGURATION
# ==========================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "4096"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

MAX_REVIEW_ATTEMPTS = int(os.getenv("MAX_REVIEW_ATTEMPTS", "3"))
MAX_FILES_TO_SCAN = int(os.getenv("MAX_FILES_TO_SCAN", "1500"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(2 * 1024 * 1024)))
MAX_LLM_FILE_BYTES = int(os.getenv("MAX_LLM_FILE_BYTES", str(350 * 1024)))

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "bower_components",
    "vendor",
    "dist",
    "build",
    "out",
    "target",
    "coverage",
    ".next",
    ".nuxt",
    ".turbo",
    ".gradle",
    "bin",
    "obj",
    "Pods",
    "DerivedData",
}

TEXT_EXTENSIONS = {
    ".py", ".pyw", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala", ".go", ".rs", ".c", ".h", ".cc", ".cpp",
    ".cxx", ".hpp", ".cs", ".fs", ".fsx", ".php", ".rb", ".rake", ".swift",
    ".m", ".mm", ".dart", ".lua", ".r", ".R", ".sh", ".bash", ".zsh", ".ps1",
    ".sql", ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
    ".xml", ".svg", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".md", ".txt", ".env", ".properties", ".gradle",
}

LANGUAGE_BY_EXT = {
    ".py": "Python", ".pyw": "Python", ".pyi": "Python", ".js": "JavaScript",
    ".jsx": "JavaScript/JSX", ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript/TSX", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".scala": "Scala", ".go": "Go", ".rs": "Rust", ".c": "C", ".h": "C/C++ Header",
    ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++ Header", ".cs": "C#",
    ".fs": "F#", ".fsx": "F#", ".php": "PHP", ".rb": "Ruby", ".rake": "Ruby",
    ".swift": "Swift", ".m": "Objective-C", ".mm": "Objective-C++", ".dart": "Dart",
    ".lua": "Lua", ".r": "R", ".R": "R", ".sh": "Shell", ".bash": "Shell",
    ".zsh": "Shell", ".ps1": "PowerShell", ".sql": "SQL", ".html": "HTML",
    ".htm": "HTML", ".css": "CSS", ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
    ".vue": "Vue", ".svelte": "Svelte",
}


# ==========================================
# 1. STATE
# ==========================================
class AgentState(TypedDict, total=False):
    project_address: str
    user_request: str
    requirements: str

    manifest: dict[str, Any]
    manifest_json: str

    plan_stack: list[dict[str, Any]]
    current_task: dict[str, Any] | None

    issue_report: str
    review_attempts: int

    iterations: int
    file_attempts: dict[str, int]
    task_attempts: dict[str, int]

    run_dir: Path
    start_time: float
    elapsed_seconds: float
    task_durations: dict[str, float]
    task_history: list[dict[str, Any]]

    last_edit: dict[str, Any]
    evaluation: dict[str, Any]
    total_tasks: int


# ==========================================
# 2. MODEL / LOGGING UTILITIES
# ==========================================

def build_llm() -> ChatOllama:
    kwargs: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "num_predict": OLLAMA_NUM_PREDICT,
        "num_ctx": OLLAMA_NUM_CTX,
        "temperature": OLLAMA_TEMPERATURE,
    }
    if OLLAMA_BASE_URL:
        kwargs["base_url"] = OLLAMA_BASE_URL
    return ChatOllama(**kwargs)


llm = build_llm()


def write_log(run_dir: Path, step: str, details: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "execution_log.txt", "a", encoding="utf-8") as handle:
        timestamp = datetime.now().strftime("%H:%M:%S")
        handle.write(f"\n[{timestamp}] === {step.upper()} ===\n")
        handle.write(str(details) + "\n")
        handle.write("-" * 100 + "\n")


def safe_json_load(raw: str) -> Any:
    cleaned = raw.strip().replace("\ufeff", "")
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z0-9_+.-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidates = [cleaned.find("{"), cleaned.find("[")]
        candidates = [position for position in candidates if position >= 0]
        if not candidates:
            raise ValueError("LLM output did not contain JSON.")
        start = min(candidates)
        for end in range(len(cleaned), start, -1):
            try:
                return json.loads(cleaned[start:end].strip())
            except json.JSONDecodeError:
                continue
    raise ValueError("Unable to parse JSON from LLM response.")


def read_text_file(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"Binary file cannot be edited as text: {path}")
    text = data.decode("utf-8")
    return text if limit is None else text[:limit]


def is_probably_text(path: Path) -> bool:
    if path.suffix in TEXT_EXTENSIONS:
        return True
    return path.name in {
        ".gitignore", ".dockerignore", ".editorconfig", "Dockerfile", "Makefile",
    }


def relative_project_path(project_root: Path, path: Path) -> str:
    return path.relative_to(project_root).as_posix()


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Target path escapes project root: {relative_path}") from exc
    return candidate


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def backup_file(project_root: Path, run_dir: Path, path: Path) -> str | None:
    if not path.exists():
        return None
    rel = relative_project_path(project_root, path)
    backup_dir = run_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = backup_dir / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination.relative_to(run_dir).as_posix()


def run_optional_command(command: list[str], cwd: Path, timeout: int = 60) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return completed.returncode == 0, (completed.stdout + "\n" + completed.stderr).strip()


# ==========================================
# 3. STATIC REPOSITORY ANALYSIS
# ==========================================

def detect_symbols(text: str, language: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    lines = text.splitlines()

    def add(name: str, kind: str, line: int) -> None:
        if name:
            symbols.append({"name": name, "kind": kind, "line": line})

    if language == "Python":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(node.name, "function", node.lineno)
                elif isinstance(node, ast.ClassDef):
                    add(node.name, "class", node.lineno)
            return symbols[:250]
        except SyntaxError:
            pass

    patterns: list[tuple[str, str]] = []
    if language in {"JavaScript", "JavaScript/JSX", "TypeScript", "TypeScript/TSX"}:
        patterns = [
            (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", "function"),
            (r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", "function"),
            (r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
        ]
    elif language in {"Java", "Kotlin", "Scala", "C#", "F#", "PHP"}:
        patterns = [
            (r"^\s*(?:public|private|protected|internal|static|final|abstract|suspend|async|override|\s)*"
             r"(?:[\w<>\[\],.?]+)\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*(?:public|private|protected|internal|static|abstract)?\s*class\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Go":
        patterns = [
            (r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*type\s+([A-Za-z_]\w*)\s+struct", "class"),
        ]
    elif language == "Rust":
        patterns = [
            (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)", "class"),
            (r"^\s*(?:pub\s+)?trait\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language in {"C", "C/C++ Header", "C++", "C++ Header", "Objective-C", "Objective-C++"}:
        patterns = [
            (r"^\s*(?:[\w:*&<>\[\],\s]+)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?", "function"),
            (r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Swift":
        patterns = [
            (r"^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|static\s+|mutating\s+)*func\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*(?:class|struct|protocol|enum)\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Ruby":
        patterns = [
            (r"^\s*def\s+([A-Za-z_]\w*[!?=]?)", "function"),
            (r"^\s*class\s+([A-Za-z_]\w*)", "class"),
            (r"^\s*module\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Dart":
        patterns = [
            (r"^\s*(?:Future<[^>]+>|[\w<>?]+)\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*class\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Lua":
        patterns = [(r"^\s*function\s+([A-Za-z_][\w.]*)\s*\(", "function")]
    elif language == "Shell":
        patterns = [
            (r"^\s*([A-Za-z_]\w*)\s*\(\s*\)\s*\{", "function"),
            (r"^\s*function\s+([A-Za-z_]\w*)", "function"),
        ]

    for line_number, line in enumerate(lines, 1):
        for pattern, kind in patterns:
            match = re.search(pattern, line)
            if match:
                add(match.group(1), kind, line_number)
                break
    return symbols[:250]


def detect_frameworks(project_root: Path, relative_files: set[str]) -> list[str]:
    detected: set[str] = set()

    def load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(read_text_file(path, MAX_FILE_BYTES))
        except Exception:
            return {}

    package = load_json(project_root / "package.json")
    dependencies = {}
    dependencies.update(package.get("dependencies", {}))
    dependencies.update(package.get("devDependencies", {}))
    node_map = {
        "react": "React", "next": "Next.js", "vue": "Vue", "@angular/core": "Angular",
        "svelte": "Svelte", "express": "Express", "@nestjs/core": "NestJS", "electron": "Electron",
        "vite": "Vite", "webpack": "Webpack", "tailwindcss": "Tailwind CSS",
    }
    detected.update(node_map[key] for key in node_map if key in dependencies)

    python_dependencies: set[str] = set()
    for name in ("requirements.txt", "requirements-dev.txt"):
        path = project_root / name
        if path.exists():
            try:
                for line in read_text_file(path, MAX_FILE_BYTES).splitlines():
                    value = re.split(r"[<>=!~\[]", line.strip().lower(), 1)[0]
                    if value and not value.startswith("#"):
                        python_dependencies.add(value)
            except Exception:
                continue

    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            python_dependencies.update(
                value.lower()
                for value in re.findall(r"['\"]([A-Za-z0-9_.-]+)(?:[<>=!~\[])", read_text_file(pyproject, MAX_FILE_BYTES))
            )
        except Exception:
            pass

    python_map = {
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "streamlit": "Streamlit",
        "tensorflow": "TensorFlow", "torch": "PyTorch", "pytorch": "PyTorch",
        "langchain": "LangChain", "langgraph": "LangGraph",
    }
    detected.update(python_map[key] for key in python_map if key in python_dependencies)

    for filename in relative_files:
        name = Path(filename).name.lower()
        if name == "manage.py":
            detected.add("Django")
        if name == "pubspec.yaml":
            detected.add("Flutter")
        if name.endswith(".csproj") or name.endswith(".sln"):
            detected.add(".NET")

    if "pom.xml" in relative_files:
        try:
            pom = read_text_file(project_root / "pom.xml", MAX_FILE_BYTES).lower()
            if "spring-boot" in pom or "org.springframework" in pom:
                detected.add("Spring Boot/Spring Framework")
        except Exception:
            pass

    return sorted(detected)


def detect_architecture(files: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [item["path"] for item in files]
    normalized = {f"/{path.lower()}" for path in paths}
    top_level_dirs = sorted({Path(path).parts[0] for path in paths if len(Path(path).parts) > 1})
    signals: list[str] = []
    patterns = {
        "components": "/components/",
        "controllers": "/controllers/",
        "services": "/services/",
        "repositories": "/repositories/",
        "models": "/models/",
        "views": "/views/",
        "routing": "/routes/",
        "routers": "/router/",
        "tests": "/tests/",
    }
    for label, pattern in patterns.items():
        if any(pattern in path or path.endswith(pattern.rstrip("/")) for path in normalized):
            signals.append(label)

    if any(Path(path).name.lower() in {"manage.py", "settings.py", "urls.py"} for path in paths):
        signals.append("django-conventions")
    if any(Path(path).name.lower() in {"main.py", "app.py", "main.go", "main.rs"} for path in paths):
        signals.append("entrypoint")

    signal_set = set(signals)
    if {"controllers", "services", "repositories"} <= signal_set:
        style = "layered"
    elif "components" in signal_set:
        style = "component-oriented"
    elif {"models", "views"} <= signal_set:
        style = "MVC-like"
    elif "django-conventions" in signal_set:
        style = "Django-conventional"
    elif "entrypoint" in signal_set:
        style = "single-service-or-script"
    else:
        style = "repository-conventional"

    return {"style": style, "signals": signals, "top_level_directories": top_level_dirs[:100]}


def collect_repository_context(project_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    language_counts: dict[str, int] = {}
    total_bytes = 0

    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for filename in sorted(filenames):
            if len(files) >= MAX_FILES_TO_SCAN:
                break
            path = Path(root) / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES or not is_probably_text(path):
                continue
            try:
                raw = path.read_bytes()
                if b"\x00" in raw:
                    continue
                text = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            relative = relative_project_path(project_root, path)
            language = LANGUAGE_BY_EXT.get(path.suffix, "Unknown")
            if filename in {"Dockerfile", "Makefile"}:
                language = "Build/Config"
            files.append({
                "path": relative,
                "language": language,
                "size_bytes": size,
                "sha256": sha256_bytes(raw),
                "symbols": detect_symbols(text, language),
            })
            language_counts[language] = language_counts.get(language, 0) + 1
            total_bytes += size
        if len(files) >= MAX_FILES_TO_SCAN:
            break

    files.sort(key=lambda item: item["path"])
    relative_files = {item["path"] for item in files}
    languages = [
        {"name": name, "file_count": count}
        for name, count in sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    entrypoint_names = {
        "main.py", "app.py", "manage.py", "main.go", "main.rs", "index.js", "index.ts",
        "server.js", "server.ts", "main.java", "application.java", "program.cs", "pubspec.yaml",
    }
    entrypoints = [item["path"] for item in files if Path(item["path"]).name.lower() in entrypoint_names]

    manifest = {
        "schema_version": "2.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": {
            "address": str(project_root),
            "name": project_root.name,
            "git_repository": (project_root / ".git").exists(),
        },
        "languages": languages,
        "frameworks": detect_frameworks(project_root, relative_files),
        "architecture": detect_architecture(files),
        "entrypoints": entrypoints,
        "files": files,
        "scan_limits": {"max_files": MAX_FILES_TO_SCAN, "max_file_bytes": MAX_FILE_BYTES},
        "total_files_scanned": len(files),
        "total_text_bytes_scanned": total_bytes,
    }
    return manifest


def language_context(manifest: dict[str, Any], target_file: str) -> str:
    target = next((item for item in manifest.get("files", []) if item["path"] == target_file), None)
    primary = ", ".join(item["name"] for item in manifest.get("languages", [])[:8]) or "Unknown"
    frameworks = ", ".join(manifest.get("frameworks", [])) or "None detected"
    architecture = manifest.get("architecture", {})
    return "\n".join([
        f"Project languages: {primary}",
        f"Frameworks/libraries detected: {frameworks}",
        f"Architecture style: {architecture.get('style', 'unknown')}",
        f"Architecture signals: {', '.join(architecture.get('signals', [])) or 'none'}",
        f"Target file language: {(target or {}).get('language', 'Unknown')}",
    ])


# ==========================================
# 4. REQUIREMENT NODE
# ==========================================

def requirement_node(state: AgentState) -> AgentState:
    project = Path(state["project_address"]).expanduser().resolve()
    request = state["user_request"].strip()
    if not project.is_dir():
        raise ValueError(f"project_address is not a directory: {project}")
    if not request:
        raise ValueError("user_request must not be empty.")

    run_dir = state.get("run_dir") or (Path.cwd() / "SDLC_Runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = f"""
You are a senior software requirements engineer working on an EXISTING local repository.

PROJECT:
{project}

USER REQUEST:
{request}

Produce implementation requirements only. Do not invent unrelated features.
Return JSON with exactly these keys:
{{
  "project_goal": "one concise sentence",
  "functional_requirements": ["..."],
  "non_functional_requirements": ["..."],
  "constraints": ["..."],
  "acceptance_criteria": ["..."]
}}
Be concrete enough for another agent to edit an existing codebase safely.
"""
    write_log(run_dir, "Requirement Node - PROMPT", prompt)
    response = llm.invoke(prompt)
    requirements = getattr(response, "content", str(response)).strip()
    write_log(run_dir, "Requirement Node - OUTPUT", requirements)
    (run_dir / "requirements.txt").write_text(requirements, encoding="utf-8")

    return {
        "project_address": str(project),
        "user_request": request,
        "requirements": requirements,
        "run_dir": run_dir,
        "start_time": state.get("start_time", time.perf_counter()),
        "issue_report": "",
        "review_attempts": 0,
        "iterations": state.get("iterations", 0),
        "file_attempts": state.get("file_attempts", {}),
        "task_attempts": state.get("task_attempts", {}),
        "task_durations": state.get("task_durations", {}),
        "task_history": state.get("task_history", []),
        "last_edit": {},
        "evaluation": {},
        "total_tasks": 0,
    }


# ==========================================
# 5. ANALYZER NODE
# ==========================================

def analyzer_node(state: AgentState) -> AgentState:
    project = Path(state["project_address"]).resolve()
    run_dir = Path(state["run_dir"])
    print("\n🔬 [Analyzer] Scanning repository...")

    manifest = collect_repository_context(project)
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    (run_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    write_log(run_dir, "Analyzer Node - MANIFEST", manifest_json[:50000])
    return {"manifest": manifest, "manifest_json": manifest_json}


# ==========================================
# 6. PLANNER NODE
# ==========================================

def planner_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    prompt = f"""
You are a senior software architect planning SURGICAL edits to an EXISTING repository.

USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

REPOSITORY MANIFEST:
{state["manifest_json"]}

Create a granular execution plan as a LIFO STACK.

STRICT STACK RULE:
- Return tasks in bottom-to-top stack order.
- The LAST array element is the FIRST task the Editor executes with pop().
- Every task must be independently executable and reviewable.
- One task should normally touch ONE target file.
- Existing-file tasks MUST specify the exact function, class, method, handler, selector, block, or symbol to change.
- If no symbol exists, use a precise section such as "module-level imports" or "configuration object".
- New files must use action=create and include their relative path.
- Do not include unrelated cleanup, formatting-only work, or speculative refactors.
- Prefer several small tasks over one large task.

Return ONLY JSON:
{{
  "plan_stack": [
    {{
      "id": "T01",
      "action": "modify|create",
      "file": "relative/path.ext",
      "function": "exact symbol or precise section",
      "change": "exact intended change",
      "rationale": "why this edit is required",
      "new_file": false,
      "acceptance_criteria": ["specific checks"],
      "related_files": ["relative/path.ext"]
    }}
  ]
}}
"""
    write_log(run_dir, "Planner Node - PROMPT", prompt)
    response = llm.invoke(prompt)
    raw = getattr(response, "content", str(response)).strip()
    write_log(run_dir, "Planner Node - RAW OUTPUT", raw)
    parsed = safe_json_load(raw)
    tasks = parsed.get("plan_stack") if isinstance(parsed, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner did not return a non-empty plan_stack.")

    known_files = {item["path"] for item in state["manifest"].get("files", [])}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_task in enumerate(tasks, 1):
        if not isinstance(raw_task, dict):
            raise ValueError(f"Planner task #{index} is not an object.")
        task = deepcopy(raw_task)
        task_id = str(task.get("id") or f"T{index:02d}")
        action = str(task.get("action", "")).strip().lower()
        file_path = str(task.get("file", "")).strip().replace("\\", "/")
        function = str(task.get("function", "")).strip()
        change = str(task.get("change", "")).strip()
        if task_id in seen:
            raise ValueError(f"Duplicate planner task id: {task_id}")
        if action not in {"modify", "create"}:
            raise ValueError(f"Task {task_id}: action must be modify or create.")
        if not file_path or Path(file_path).is_absolute():
            raise ValueError(f"Task {task_id}: invalid relative file path.")
        if not function or not change:
            raise ValueError(f"Task {task_id}: function/section and change are required.")
        if action == "modify" and file_path not in known_files:
            raise ValueError(f"Task {task_id}: modify target is absent from manifest: {file_path}")
        task.update({
            "id": task_id,
            "action": action,
            "file": Path(file_path).as_posix(),
            "function": function,
            "change": change,
            "new_file": action == "create",
            "acceptance_criteria": list(task.get("acceptance_criteria", [])),
            "related_files": list(task.get("related_files", [])),
        })
        normalized.append(task)
        seen.add(task_id)

    plan_json = json.dumps(normalized, indent=2, ensure_ascii=False)
    (run_dir / "plan_stack.json").write_text(plan_json, encoding="utf-8")
    write_log(run_dir, "Planner Node - STACK", plan_json)
    return {"plan_stack": normalized, "current_task": None, "issue_report": "", "review_attempts": 0, "total_tasks": len(normalized)}


# ==========================================
# 7. EDIT OPERATION VALIDATION/APPLICATION
# ==========================================

def sanitize_editor_output(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Editor response must be a JSON object.")
    action = parsed.get("action")
    file_path = str(parsed.get("file", "")).strip().replace("\\", "/")
    if action not in {"modify", "create"}:
        raise ValueError("Editor action must be modify or create.")
    if not file_path or Path(file_path).is_absolute():
        raise ValueError("Editor response contains an invalid relative file path.")

    if action == "create":
        content = parsed.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Create-file response must contain non-empty content.")
        return {"action": "create", "file": Path(file_path).as_posix(), "content": content, "operations": []}

    operations = parsed.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Modify-file response must contain operations.")
    clean: list[dict[str, Any]] = []
    for index, operation in enumerate(operations, 1):
        if not isinstance(operation, dict):
            raise ValueError(f"Edit operation #{index} is not an object.")
        search = operation.get("search")
        replace = operation.get("replace")
        expected = operation.get("expected_occurrences", 1)
        if not isinstance(search, str) or not search:
            raise ValueError(f"Edit operation #{index} requires non-empty search.")
        if not isinstance(replace, str):
            raise ValueError(f"Edit operation #{index} requires string replace.")
        if not isinstance(expected, int) or expected < 1:
            raise ValueError(f"Edit operation #{index} expected_occurrences must be positive.")
        clean.append({"search": search, "replace": replace, "expected_occurrences": expected})
    return {"action": "modify", "file": Path(file_path).as_posix(), "content": None, "operations": clean}


def apply_edit_plan(project_root: Path, run_dir: Path, task: dict[str, Any], edit_plan: dict[str, Any]) -> dict[str, Any]:
    expected_file = Path(task["file"]).as_posix()
    if edit_plan["file"] != expected_file:
        raise ValueError(f"Editor targeted {edit_plan['file']} but task targets {expected_file}.")

    target = resolve_repo_path(project_root, expected_file)
    if edit_plan["action"] == "create":
        if target.exists():
            raise FileExistsError(f"Task declares new file but it exists: {expected_file}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit_plan["content"], encoding="utf-8")
        return {
            "action": "create",
            "file": expected_file,
            "backup": None,
            "operations": [],
            "before_sha256": None,
            "after_sha256": sha256_bytes(target.read_bytes()),
        }

    if not target.exists():
        raise FileNotFoundError(f"Target file does not exist: {expected_file}")
    before = read_text_file(target)
    backup = backup_file(project_root, run_dir, target)
    updated = before
    operation_results = []
    for index, operation in enumerate(edit_plan["operations"], 1):
        search = operation["search"]
        replace = operation["replace"]
        expected = operation["expected_occurrences"]
        count = updated.count(search)
        if count != expected:
            raise ValueError(
                f"Operation #{index} expected {expected} occurrence(s), found {count} in {expected_file}."
            )
        updated = updated.replace(search, replace, expected)
        operation_results.append({
            "index": index,
            "expected_occurrences": expected,
            "matched_occurrences": count,
            "search_preview": search[:240],
            "replace_preview": replace[:240],
        })

    if updated == before:
        raise ValueError("Editor produced no file change.")
    target.write_text(updated, encoding="utf-8")
    return {
        "action": "modify",
        "file": expected_file,
        "backup": backup,
        "operations": operation_results,
        "before_sha256": sha256_bytes(before.encode("utf-8")),
        "after_sha256": sha256_bytes(updated.encode("utf-8")),
    }


# ==========================================
# 8. EDITOR NODE
# ==========================================

def editor_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    project_root = Path(state["project_address"])
    manifest = state["manifest"]
    plan_stack = list(state.get("plan_stack", []))
    current_task = deepcopy(state.get("current_task")) if state.get("current_task") else None
    issue_report = state.get("issue_report", "")

    # After a PASS, this re-entry clears the completed task and pops exactly one next task.
    if current_task and current_task.get("review_status") == "PASS":
        current_task = None
        issue_report = ""

    if current_task is None:
        if not plan_stack:
            return {"plan_stack": plan_stack, "current_task": None, "issue_report": ""}
        current_task = deepcopy(plan_stack.pop())

    task_id = current_task["id"]
    attempts_map = dict(state.get("task_attempts", {}))
    attempt = attempts_map.get(task_id, 0) + 1
    attempts_map[task_id] = attempt

    target_file = current_task["file"]
    target = resolve_repo_path(project_root, target_file)
    if target.exists() and target.stat().st_size > MAX_LLM_FILE_BYTES:
        raise ValueError(f"Target file exceeds MAX_LLM_FILE_BYTES: {target_file}")
    current_code = read_text_file(target) if target.exists() else ""

    target_manifest = next((item for item in manifest.get("files", []) if item["path"] == target_file), {})
    system_prompt = f"""
You are the repository Editor Agent in a production code-editing pipeline.

LANGUAGE / FRAMEWORK CONTEXT:
{language_context(manifest, target_file)}

TARGET SYMBOL CONTEXT:
{json.dumps(target_manifest.get("symbols", []), indent=2, ensure_ascii=False)}

TASK:
{json.dumps(current_task, indent=2, ensure_ascii=False)}

OVERARCHING USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

PREVIOUS REVIEW ISSUE:
{issue_report or "None"}

CURRENT FILE ({target_file}):
```text
{current_code}
```

EDITING RULES:
1. Work inside the existing repository and preserve its architecture/conventions.
2. For an existing file, NEVER rewrite the entire file.
3. Existing files may ONLY be changed with exact literal search-and-replace operations.
4. Every search string must be an exact substring from the CURRENT FILE.
5. expected_occurrences must equal the exact number of occurrences to replace.
6. Keep every unrelated line unchanged.
7. Do not perform unrelated refactors, formatting, renaming, or cleanup.
8. For a NEW file only, return complete content.
9. Never target a path outside the repository.
10. Never use regex, line numbers, placeholders, or markdown outside JSON.

Return ONLY JSON.
Existing file schema:
{{
  "action": "modify",
  "file": "{target_file}",
  "operations": [
    {{
      "search": "exact existing substring",
      "replace": "replacement substring",
      "expected_occurrences": 1
    }}
  ]
}}

New file schema:
{{
  "action": "create",
  "file": "{target_file}",
  "content": "complete new file content"
}}
"""
    write_log(run_dir, f"Editor Node [{task_id}] - PROMPT", system_prompt)

    started = time.perf_counter()
    response = llm.invoke(system_prompt)
    raw = getattr(response, "content", str(response)).strip()
    write_log(run_dir, f"Editor Node [{task_id}] - RAW OUTPUT", raw[:30000])

    try:
        edit_plan = sanitize_editor_output(safe_json_load(raw))
        if edit_plan["action"] != current_task["action"]:
            raise ValueError(
                f"Editor action {edit_plan['action']} conflicts with planner action {current_task['action']}."
            )
        result = apply_edit_plan(project_root, run_dir, current_task, edit_plan)
    except Exception as exc:
        duration = time.perf_counter() - started
        issue = (
            f"# Editor Issue — {task_id}\n\n"
            f"## Problem\n{type(exc).__name__}: {exc}\n\n"
            f"## Required Action\n"
            f"Regenerate a surgical edit plan for `{target_file}`.\n"
        )
        (run_dir / "issue.md").write_text(issue, encoding="utf-8")
        write_log(run_dir, f"Editor Node [{task_id}] - APPLY ERROR", issue)
        return {
            "plan_stack": plan_stack,
            "current_task": current_task,
            "issue_report": issue,
            "review_attempts": state.get("review_attempts", 0) + 1,
            "iterations": state.get("iterations", 0) + 1,
            "task_attempts": attempts_map,
            "last_edit": {},
            "task_durations": {**state.get("task_durations", {}), task_id: round(state.get("task_durations", {}).get(task_id, 0.0) + duration, 4)},
        }

    duration = time.perf_counter() - started
    file_attempts = dict(state.get("file_attempts", {}))
    file_attempts[target_file] = file_attempts.get(target_file, 0) + 1
    current_task["review_status"] = "PENDING"
    current_task["last_edit_seconds"] = round(duration, 4)

    history = list(state.get("task_history", []))
    history.append({
        "task_id": task_id,
        "attempt": attempt,
        "file": target_file,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 4),
        "edit": result,
    })
    return {
        "plan_stack": plan_stack,
        "current_task": current_task,
        "issue_report": "",
        "review_attempts": 0,
        "iterations": state.get("iterations", 0) + 1,
        "file_attempts": file_attempts,
        "task_attempts": attempts_map,
        "task_durations": {**state.get("task_durations", {}), task_id: round(state.get("task_durations", {}).get(task_id, 0.0) + duration, 4)},
        "task_history": history,
        "last_edit": result,
    }


# ==========================================
# 9. REVIEWER NODE
# ==========================================

def local_validation(project_root: Path, target_file: str) -> tuple[bool, str]:
    target = resolve_repo_path(project_root, target_file)
    if not target.exists():
        return False, f"Target file does not exist: {target_file}"
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"Target file is not valid UTF-8 text: {target_file}"

    language = LANGUAGE_BY_EXT.get(target.suffix, "")
    if language == "Python":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return False, f"Python syntax error: {exc}"

    if target.suffix.lower() in {".json", ".jsonc"}:
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            return False, f"JSON syntax error: {exc}"

    if target.suffix.lower() in {".js", ".mjs", ".cjs", ".jsx"}:
        node = shutil.which("node")
        if node:
            ok, output = run_optional_command([node, "--check", str(target)], project_root, 30)
            if not ok:
                return False, f"Node syntax check failed: {output}"

    return True, "Local syntax/structure validation passed."


def reviewer_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    project_root = Path(state["project_address"])
    task = state.get("current_task")
    if not task:
        return {}

    target_file = task["file"]
    target_path = resolve_repo_path(project_root, target_file)
    current_code = read_text_file(target_path, MAX_LLM_FILE_BYTES)
    local_ok, local_feedback = local_validation(project_root, target_file)
    if not local_ok:
        issue = (
            f"# Review Issue — {task['id']}\n\n"
            f"## Local Validation\n{local_feedback}\n\n"
            f"## Required Surgical Fix\nFix only task `{task['id']}` in `{target_file}`.\n"
        )
        (run_dir / "issue.md").write_text(issue, encoding="utf-8")
        write_log(run_dir, f"Reviewer Node [{task['id']}] - LOCAL FAIL", issue)
        return {
            "issue_report": issue,
            "review_attempts": state.get("review_attempts", 0) + 1,
            "current_task": task,
        }

    related: list[str] = []
    for relative in task.get("related_files", [])[:8]:
        try:
            path = resolve_repo_path(project_root, relative)
            if path.exists() and path.is_file() and path.stat().st_size <= MAX_LLM_FILE_BYTES:
                related.append(f"\n--- {relative} ---\n{read_text_file(path, MAX_LLM_FILE_BYTES)}")
        except Exception:
            continue

    prompt = f"""
You are a senior code reviewer validating one SURGICAL repository edit.

OVERARCHING USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

REPOSITORY MANIFEST:
{state["manifest_json"]}

EXECUTION TASK:
{json.dumps(task, indent=2, ensure_ascii=False)}

TARGET FILE:
{target_file}

CURRENT TARGET CODE:
{current_code}

RELATED FILE CONTEXT:
{"".join(related) if related else "None"}

LOCAL VALIDATION:
{local_feedback}

CHECK:
1. Exact task satisfaction.
2. Requested function/class/section is correctly changed.
3. Language/framework/architecture consistency.
4. Obvious syntax, runtime, API, reference, import, type, selector, naming, or path errors.
5. No unrelated behavior was damaged.
6. Acceptance criteria are satisfied.
7. Do not request unrelated improvements.

If fully correct, return ONLY:
PASS

Otherwise return ONLY JSON:
{{
  "status": "FAIL",
  "summary": "concise issue",
  "replace": "exact problematic snippet or statement",
  "with": "exact corrected snippet or statement",
  "reason": "why this violates task/manifest/request"
}}
"""
    write_log(run_dir, f"Reviewer Node [{task['id']}] - PROMPT", prompt)
    response = llm.invoke(prompt)
    review_output = getattr(response, "content", str(response)).strip()
    write_log(run_dir, f"Reviewer Node [{task['id']}] - OUTPUT", review_output)

    if re.match(r"^\s*PASS\.?\s*$", review_output, re.IGNORECASE):
        approved = deepcopy(task)
        approved["review_status"] = "PASS"
        approved["approved_at"] = datetime.now().isoformat(timespec="seconds")
        return {"issue_report": "", "review_attempts": 0, "current_task": approved}

    try:
        structured = safe_json_load(review_output)
        if not isinstance(structured, dict):
            raise ValueError("Reviewer did not return a JSON object.")
        issue = (
            f"# Review Issue — {task['id']}\n\n"
            f"## Summary\n{structured.get('summary', 'Reviewer found an issue.')}\n\n"
            f"## Surgical Change\n"
            f"**Replace:**\n```text\n{structured.get('replace', '')}\n```\n\n"
            f"**With:**\n```text\n{structured.get('with', '')}\n```\n\n"
            f"## Reason\n{structured.get('reason', '')}\n"
        )
    except Exception:
        issue = f"# Review Issue — {task['id']}\n\n{review_output}\n"

    (run_dir / "issue.md").write_text(issue, encoding="utf-8")
    return {
        "issue_report": issue,
        "review_attempts": state.get("review_attempts", 0) + 1,
        "current_task": task,
    }


# ==========================================
# 10. REVIEW ROUTING
# ==========================================

def route_review(state: AgentState) -> str:
    if state.get("issue_report"):
        return "abort" if state.get("review_attempts", 0) >= MAX_REVIEW_ATTEMPTS else "fail"
    return "pass"


def route_editor(state: AgentState) -> str:
    # Application/parsing failures retry the same task directly.
    if state.get("issue_report"):
        return "retry"
    # When the Editor re-enters after a PASS and the stack is empty, evaluate.
    if not state.get("current_task") and not state.get("plan_stack"):
        return "evaluator"
    return "review"


# ==========================================
# 11. EVALUATION (DeepEval + timing/task matrix)
# ==========================================
if DEEPEVAL_AVAILABLE:

    class OllamaEvalModel(DeepEvalBaseLLM):
        def __init__(self, chat_model: ChatOllama, name: str):
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
    if not DEEPEVAL_AVAILABLE:
        return []
    judge = OllamaEvalModel(llm, OLLAMA_MODEL)
    return [
        GEval(
            name="Correctness",
            criteria=(
                "Using requirements, manifest, and task history as ground truth, determine whether the final "
                "changed repository state plausibly implements the requested behavior without obvious runtime, "
                "API, reference, or logic bugs."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        ),
        GEval(
            name="Completeness",
            criteria=(
                "Using requirements and manifest as specification, determine whether the final repository state "
                "implements the requested functionality with real logic and without TODO-only or empty stubs."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        ),
    ]


def deepeval_score_file(filename: str, code: str, context: str, metrics: list):
    if not DEEPEVAL_AVAILABLE or not code.strip() or not metrics:
        return None
    test_case = LLMTestCase(input=f"FILE: {filename}\n\n{context}", actual_output=code)
    results = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            results[metric.name] = {
                "score": round(metric.score, 2) if metric.score is not None else None,
                "reason": metric.reason,
            }
        except Exception as exc:
            results[metric.name] = {"score": None, "reason": f"DeepEval metric failed: {exc}"}
    return results


def compute_custom_score(attempts: int) -> float:
    return round(max(0.25, 1.0 - 0.25 * (attempts - 1)), 2)


def render_evaluation_markdown(overall: dict, report: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"**Custom score avg:** {overall['custom_score_avg']} / 1.0",
        f"**Total tasks:** {overall['total_tasks']}",
        f"**Tasks completed:** {overall['tasks_completed']}",
        f"**Time taken:** {overall['elapsed_seconds']:.3f} seconds",
        f"**Time per task:** {overall['time_per_task_seconds']:.3f} seconds",
    ]
    if "deepeval_avg_score" in overall:
        lines.append(f"**DeepEval score avg:** {overall['deepeval_avg_score']} / 1.0")
    if not overall["deepeval_available"]:
        lines.append("\n_DeepEval isn't installed (`pip install deepeval`) — showing custom scores only._")
    lines.append("")
    for filename, data in report.items():
        lines.append(f"## {filename}")
        lines.append(f"- Attempts: {data['attempts']} | Custom score: {data['custom_score']}")
        if data.get("deepeval"):
            for name, result in data["deepeval"].items():
                score = result["score"] if result["score"] is not None else "n/a"
                lines.append(f"  - **{name}**: {score} — {result['reason']}")
        lines.append("")
    lines.append("## Task Timing")
    for task in overall.get("task_timing", []):
        lines.append(
            f"- **{task['task_id']}** — {task['file']} — {task['duration_seconds']:.3f}s — "
            f"{task['attempts']} attempt(s)"
        )
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
    with open(benchmark_history_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return load_benchmark_history()


def render_benchmark_svg(records: list) -> str:
    width, height, pad = 760, 320, 48
    n = len(records)
    if n == 0:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="120">'
            f'<text x="20" y="60" font-family="sans-serif" font-size="13">No runs tracked yet.</text></svg>'
        )

    max_time = max((float(record.get("elapsed_seconds", 0.0)) for record in records), default=1.0)
    max_time = max(max_time, 1.0)

    def x_for(index: int) -> float:
        return pad + (index / max(n - 1, 1)) * (width - 2 * pad)

    def y_score(score: float) -> float:
        return height - pad - score * (height - 2 * pad)

    def y_time(seconds: float) -> float:
        return height - pad - (seconds / max_time) * (height - 2 * pad)

    score_points = [(i, float(record["custom_score_avg"])) for i, record in enumerate(records) if record.get("custom_score_avg") is not None]
    time_points = [(i, float(record["elapsed_seconds"])) for i, record in enumerate(records) if record.get("elapsed_seconds") is not None]
    score_poly = " ".join(f"{x_for(i):.1f},{y_score(score):.1f}" for i, score in score_points)
    time_poly = " ".join(f"{x_for(i):.1f},{y_time(seconds):.1f}" for i, seconds in time_points)

    score_svg = f'<polyline points="{score_poly}" fill="none" stroke="#16a34a" stroke-width="2"/>' if score_poly else ""
    time_svg = f'<polyline points="{time_poly}" fill="none" stroke="#7c3aed" stroke-width="2"/>' if time_poly else ""
    score_dots = "".join(
        f'<circle cx="{x_for(i):.1f}" cy="{y_score(score):.1f}" r="3" fill="#16a34a"><title>run {i + 1}: score {score:.2f}</title></circle>'
        for i, score in score_points
    )
    time_dots = "".join(
        f'<circle cx="{x_for(i):.1f}" cy="{y_time(seconds):.1f}" r="3" fill="#7c3aed"><title>run {i + 1}: {seconds:.2f}s</title></circle>'
        for i, seconds in time_points
    )
    grid = "".join(
        f'<line x1="{pad}" y1="{y_score(g):.1f}" x2="{width - pad}" y2="{y_score(g):.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        f'<text x="4" y="{y_score(g) + 4:.1f}" font-size="10" fill="#6b7280">{g:.1f}</text>'
        for g in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    score_avg = round(sum(v for _, v in score_points) / len(score_points), 2) if score_points else None
    time_avg = round(sum(v for _, v in time_points) / len(time_points), 2) if time_points else None
    legend = (
        f'<circle cx="{pad}" cy="16" r="4" fill="#16a34a"/><text x="{pad + 10}" y="20" font-size="11" fill="#374151">'
        f'Custom avg{f" ({score_avg:.2f})" if score_avg is not None else ""}</text>'
        f'<circle cx="{pad + 190}" cy="16" r="4" fill="#7c3aed"/><text x="{pad + 200}" y="20" font-size="11" fill="#374151">'
        f'Time avg{f" ({time_avg:.2f}s)" if time_avg is not None else ""}</text>'
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif">
<rect width="{width}" height="{height}" fill="white"/>
{grid}
{score_svg}{score_dots}
{time_svg}{time_dots}
{legend}
<text x="{pad}" y="{height - 8}" font-size="11" fill="#374151">Run 1</text>
<text x="{width - pad - 40}" y="{height - 8}" font-size="11" fill="#374151">Run {n}</text>
</svg>'''


def render_benchmark_html(records: list) -> str:
    deval = [r["deepeval_avg_score"] for r in records if r.get("deepeval_avg_score") is not None]
    custom = [r["custom_score_avg"] for r in records if r.get("custom_score_avg") is not None]
    elapsed = [r["elapsed_seconds"] for r in records if r.get("elapsed_seconds") is not None]
    tasks = [r["total_tasks"] for r in records if r.get("total_tasks") is not None]
    avg_deval = round(sum(deval) / len(deval), 2) if deval else "n/a"
    avg_custom = round(sum(custom) / len(custom), 2) if custom else "n/a"
    avg_time = round(sum(elapsed) / len(elapsed), 2) if elapsed else "n/a"
    avg_tasks = round(sum(tasks) / len(tasks), 2) if tasks else "n/a"
    rows = []
    for i, record in enumerate(records):
        cfg = record.get("llm_config", {})
        rows.append(
            f"<tr><td>{i + 1}</td><td>{record.get('timestamp', '')}</td>"
            f"<td>{cfg.get('model', '')}</td><td>{cfg.get('num_ctx', '')}</td>"
            f"<td>{record.get('project_name', '')}</td><td>{record.get('total_tasks', '')}</td>"
            f"<td>{record.get('elapsed_seconds', 'n/a')}</td><td>{record.get('time_per_task_seconds', 'n/a')}</td>"
            f"<td>{record.get('custom_score_avg', 'n/a')}</td><td>{record.get('deepeval_avg_score', 'n/a')}</td></tr>"
        )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Loop Workflow — Benchmark History</title>
<style>body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111827; max-width: 1200px; }} table {{ border-collapse: collapse; margin-top: 1rem; width: 100%; }} th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; font-size: 13px; }} th {{ background: #f9fafb; }}</style>
</head><body><h1>Benchmark History</h1>
<p>{len(records)} run(s) tracked — all-time avg custom score: <b>{avg_custom}</b> | all-time avg DeepEval score: <b>{avg_deval}</b> | all-time avg time: <b>{avg_time}s</b> | all-time avg task count: <b>{avg_tasks}</b></p>
{render_benchmark_svg(records)}
<table><tr><th>#</th><th>Timestamp</th><th>Model</th><th>num_ctx</th><th>Project</th><th>Total tasks</th><th>Time (s)</th><th>Time/task (s)</th><th>Custom avg</th><th>DeepEval avg</th></tr>{''.join(rows)}</table>
</body></html>"""


def evaluator_node(state: AgentState) -> AgentState:
    print("\n📊 [Evaluator] Scoring the finished repository...")
    run_dir = Path(state["run_dir"])
    project_root = Path(state["project_address"])
    start_time = state.get("start_time", time.perf_counter())
    elapsed_seconds = max(0.0, time.perf_counter() - start_time)
    changed_files = sorted({entry["file"] for entry in state.get("task_history", []) if entry.get("file")})
    total_tasks = int(state.get("total_tasks", 0))
    completed_task_ids = {entry["task_id"] for entry in state.get("task_history", [])}

    context = (
        f"USER REQUEST:\n{state['user_request']}\n\nREQUIREMENTS:\n{state['requirements']}\n\n"
        f"MANIFEST:\n{state['manifest_json']}\n\nTASK HISTORY:\n"
        f"{json.dumps(state.get('task_history', []), indent=2, ensure_ascii=False)}"
    )
    metrics = _build_deepeval_metrics() if DEEPEVAL_AVAILABLE else []
    report: dict[str, Any] = {}
    for filename in changed_files:
        try:
            path = resolve_repo_path(project_root, filename)
            code_text = read_text_file(path, MAX_LLM_FILE_BYTES) if path.exists() else ""
        except Exception as exc:
            code_text = ""
            write_log(run_dir, f"Evaluator Read Error [{filename}]", str(exc))
        attempts = state.get("file_attempts", {}).get(filename, 1)
        report[filename] = {
            "attempts": attempts,
            "custom_score": compute_custom_score(attempts),
            "deepeval": deepeval_score_file(filename, code_text, context, metrics),
        }

    custom_scores = [item["custom_score"] for item in report.values()]
    task_timing = [
        {
            "task_id": task_id,
            "file": next((entry["file"] for entry in state.get("task_history", []) if entry["task_id"] == task_id), ""),
            "duration_seconds": float(duration),
            "attempts": int(state.get("task_attempts", {}).get(task_id, 1)),
        }
        for task_id, duration in state.get("task_durations", {}).items()
    ]
    completed_count = len(completed_task_ids)
    overall = {
        "files_passed": f"{len(changed_files)}/{len(changed_files)}",
        "changed_files": changed_files,
        "total_tasks": total_tasks,
        "tasks_completed": completed_count,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "time_per_task_seconds": round(elapsed_seconds / completed_count, 4) if completed_count else 0.0,
        "task_timing": task_timing,
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "custom_score_avg": round(sum(custom_scores) / len(custom_scores), 2) if custom_scores else 0.0,
        "project_address": str(project_root),
        "project_name": state.get("manifest", {}).get("project", {}).get("name", project_root.name),
    }
    deval_scores = [
        metric["score"]
        for file_data in report.values() if file_data.get("deepeval")
        for metric in file_data["deepeval"].values()
        if metric and metric.get("score") is not None
    ]
    if deval_scores:
        overall["deepeval_avg_score"] = round(sum(deval_scores) / len(deval_scores), 2)

    evaluation = {"overall": overall, "files": report}
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "evaluation.md").write_text(render_evaluation_markdown(overall, report), encoding="utf-8")
    write_log(run_dir, "Evaluator", json.dumps(overall, indent=2, ensure_ascii=False))

    benchmark_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "project_name": overall["project_name"],
        "project_address": str(project_root),
        "user_request": state.get("user_request", "")[:300],
        "llm_config": {"model": OLLAMA_MODEL, "num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT, "temperature": OLLAMA_TEMPERATURE},
        "files_passed": overall["files_passed"],
        "total_tasks": overall["total_tasks"],
        "tasks_completed": overall["tasks_completed"],
        "elapsed_seconds": overall["elapsed_seconds"],
        "time_per_task_seconds": overall["time_per_task_seconds"],
        "custom_score_avg": overall["custom_score_avg"],
        "deepeval_available": DEEPEVAL_AVAILABLE,
        "deepeval_avg_score": overall.get("deepeval_avg_score"),
        "per_file": {filename: {"attempts": data.get("attempts")} for filename, data in report.items()},
    }
    history = append_benchmark_record(benchmark_record)
    (benchmark_history_path().parent / "benchmark_history.html").write_text(render_benchmark_html(history), encoding="utf-8")

    print(f"   Custom score avg: {overall['custom_score_avg']}/1.0")
    if "deepeval_avg_score" in overall:
        print(f"   DeepEval score avg: {overall['deepeval_avg_score']}/1.0")
    print(f"   ⏱️ Total time: {overall['elapsed_seconds']:.3f}s")
    print(f"   🧩 Total tasks: {overall['total_tasks']}")
    print(f"   ⏱️ Time/task: {overall['time_per_task_seconds']:.3f}s")
    print(f"   📈 Benchmark: {len(history)} run(s) tracked — see benchmark_history.jsonl / benchmark_history.html")
    return {"elapsed_seconds": elapsed_seconds, "evaluation": evaluation}


# ==========================================
# 12. GRAPH
# ==========================================

def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("requirements", requirement_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("requirements")
    workflow.add_edge("requirements", "analyzer")
    workflow.add_edge("analyzer", "planner")
    workflow.add_edge("planner", "editor")
    workflow.add_conditional_edges(
        "editor",
        route_editor,
        {"review": "reviewer", "retry": "editor", "evaluator": "evaluator"},
    )
    workflow.add_conditional_edges(
        "reviewer",
        route_review,
        {"fail": "editor", "pass": "editor", "abort": END},
    )
    workflow.add_edge("evaluator", END)
    return workflow.compile()


app = build_workflow()


# ==========================================
# 13. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 100)
    print("Dynamic Local Repository Editor — LangGraph + Ollama")
    print("=" * 100)

    project_address = input("\nPath to existing project/repository:\n> ").strip()
    user_request = input("\nWhat change should be made?\n> ").strip()
    project_path = Path(project_address).expanduser().resolve()
    if not project_path.is_dir():
        raise SystemExit(f"Project directory does not exist: {project_path}")
    if not user_request:
        raise SystemExit("Change request must not be empty.")

    run_dir = Path.cwd() / "SDLC_Runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    write_log(run_dir, "SYSTEM", json.dumps({
        "project_address": str(project_path),
        "user_request": user_request,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }, indent=2))

    print(f"\n🚀 Starting repository edit: {project_path}")
    print(f"📝 Request: {user_request}")
    print(f"📁 Run artifacts: {run_dir}")

    initial_state: AgentState = {
        "project_address": str(project_path),
        "user_request": user_request,
        "run_dir": run_dir,
        "start_time": start_time,
        "requirements": "",
        "manifest": {},
        "manifest_json": "",
        "plan_stack": [],
        "current_task": None,
        "issue_report": "",
        "review_attempts": 0,
        "iterations": 0,
        "file_attempts": {},
        "task_attempts": {},
        "task_durations": {},
        "task_history": [],
        "last_edit": {},
        "evaluation": {},
        "total_tasks": 0,
    }

    try:
        app.invoke(initial_state)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        failure = (
            f"# Workflow Failure\n\n**Exception:** `{type(exc).__name__}`\n\n"
            f"**Message:** {exc}\n\n**Elapsed seconds:** {elapsed:.4f}\n"
        )
        (run_dir / "workflow_failure.md").write_text(failure, encoding="utf-8")
        write_log(run_dir, "SYSTEM FAILURE", failure)
        raise

    print("\n🎉 Workflow Complete.")
    print(f"📊 Evaluation: {run_dir / 'evaluation.md'}")
    print(f"🧾 Manifest: {run_dir / 'manifest.json'}")
    print(f"📚 Plan stack: {run_dir / 'plan_stack.json'}")
    print(f"💾 Backups: {run_dir / 'backups'}")
    print(f"📈 Benchmark history: {benchmark_history_path().parent / 'benchmark_history.html'}")
