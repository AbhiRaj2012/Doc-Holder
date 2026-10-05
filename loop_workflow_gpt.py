from __future__ import annotations

import ast
import difflib
import hashlib
import html
import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# DeepEval is optional; the pipeline still runs without it.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False

# Tree-sitter is optional at runtime. The code falls back to language-aware AST/regex parsing.
try:
    from tree_sitter_language_pack import get_parser as tslp_get_parser

    TREE_SITTER_AVAILABLE = True
except Exception:
    tslp_get_parser = None
    TREE_SITTER_AVAILABLE = False

try:
    from tree_sitter_languages import get_parser as legacy_get_parser
except Exception:
    legacy_get_parser = None

# Ollama embeddings are optional. The repository editor still works with structural + lexical retrieval.
try:
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_EMBEDDINGS_AVAILABLE = True
except Exception:
    OllamaEmbeddings = None
    OLLAMA_EMBEDDINGS_AVAILABLE = False


# ==========================================
# 0. CONFIGURATION
# ==========================================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
ENABLE_RAG = os.getenv("ENABLE_RAG", "1").strip().lower() not in {"0", "false", "no", "off"}
ENABLE_LSP = os.getenv("ENABLE_LSP", "1").strip().lower() not in {"0", "false", "no", "off"}

MAX_REVIEW_ATTEMPTS = int(os.getenv("MAX_REVIEW_ATTEMPTS", "3"))
MAX_FILES_TO_SCAN = int(os.getenv("MAX_FILES_TO_SCAN", "2500"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(2 * 1024 * 1024)))
MAX_LLM_FILE_BYTES = int(os.getenv("MAX_LLM_FILE_BYTES", "12000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", str(max(7000, min(18000, int(OLLAMA_NUM_CTX * 1.8))))))
MAX_MANIFEST_PROMPT_CHARS = int(os.getenv("MAX_MANIFEST_PROMPT_CHARS", str(max(4000, min(8000, int(OLLAMA_NUM_CTX * 0.75))))))
MAX_EDIT_OPERATIONS = int(os.getenv("MAX_EDIT_OPERATIONS", "8"))
MAX_EDITOR_GENERATION_ATTEMPTS = int(os.getenv("MAX_EDITOR_GENERATION_ATTEMPTS", "3"))
MAX_PLANNER_TASKS = int(os.getenv("MAX_PLANNER_TASKS", "40"))
MAX_EDIT_SEARCH_CHARS = int(os.getenv("MAX_EDIT_SEARCH_CHARS", "12000"))
MAX_GRAPH_STEPS = int(os.getenv("MAX_GRAPH_STEPS", "80"))
RUN_PROJECT_TESTS = os.getenv("RUN_PROJECT_TESTS", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_REPO_MAP_SYMBOLS = int(os.getenv("MAX_REPO_MAP_SYMBOLS", "240"))
MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "8"))
MAX_RAG_CHUNKS = int(os.getenv("MAX_RAG_CHUNKS", "300"))
RAG_BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "16"))
RAG_CHUNK_LINES = int(os.getenv("RAG_CHUNK_LINES", "100"))
LSP_TIMEOUT_SECONDS = float(os.getenv("LSP_TIMEOUT_SECONDS", "8"))

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "__pycache__", ".tox", ".venv", "venv", "env", "node_modules", "bower_components", "vendor",
    "dist", "build", "out", "target", "coverage", ".next", ".nuxt", ".turbo", ".gradle", "bin", "obj",
    "Pods", "DerivedData", ".cache", ".parcel-cache", ".dart_tool", "cmake-build-debug", "cmake-build-release",
}

SENSITIVE_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development", ".env.test",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials", "credentials.json",
    "secrets.json", "secrets.yaml", "secrets.yml", "service-account.json",
}

TEXT_EXTENSIONS = {
    ".py", ".pyw", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java", ".kt", ".kts",
    ".scala", ".go", ".rs", ".c", ".h", ".hh", ".cc", ".cpp", ".cxx", ".hpp", ".hxx", ".cs",
    ".fs", ".fsx", ".php", ".rb", ".rake", ".swift", ".m", ".mm", ".dart", ".lua", ".r", ".R",
    ".sh", ".bash", ".zsh", ".ps1", ".sql", ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte", ".xml", ".svg", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".md", ".txt", ".properties", ".gradle", ".cmake", ".proto",
}

LANGUAGE_BY_EXT = {
    ".py": "Python", ".pyw": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript/JSX", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript/TSX",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin", ".scala": "Scala",
    ".go": "Go", ".rs": "Rust",
    ".c": "C", ".h": "C/C++ Header", ".hh": "C++ Header", ".cc": "C++", ".cpp": "C++",
    ".cxx": "C++", ".hpp": "C++ Header", ".hxx": "C++ Header", ".cs": "C#", ".fs": "F#", ".fsx": "F#",
    ".php": "PHP", ".rb": "Ruby", ".rake": "Ruby", ".swift": "Swift", ".m": "Objective-C",
    ".mm": "Objective-C++", ".dart": "Dart", ".lua": "Lua", ".r": "R", ".R": "R",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".ps1": "PowerShell", ".sql": "SQL",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
    ".vue": "Vue", ".svelte": "Svelte",
}

TREE_SITTER_LANGUAGE_KEYS = {
    "Python": "python", "JavaScript": "javascript", "JavaScript/JSX": "javascript", "TypeScript": "typescript",
    "TypeScript/TSX": "tsx", "Java": "java", "Kotlin": "kotlin", "Scala": "scala", "Go": "go", "Rust": "rust",
    "C": "c", "C++": "cpp", "C/C++ Header": "cpp", "C++ Header": "cpp", "C#": "c_sharp", "PHP": "php",
    "Ruby": "ruby", "Swift": "swift", "Objective-C": "objc", "Objective-C++": "cpp", "Dart": "dart",
    "Lua": "lua", "R": "r", "Shell": "bash", "SQL": "sql",
}

LANGUAGE_LSP_IDS = {
    "Python": "python", "JavaScript": "javascript", "JavaScript/JSX": "javascriptreact", "TypeScript": "typescript",
    "TypeScript/TSX": "typescriptreact", "C": "c", "C++": "cpp", "C/C++ Header": "cpp", "C++ Header": "cpp",
    "Go": "go", "Rust": "rust", "Java": "java", "Kotlin": "kotlin", "C#": "csharp",
}

DEFAULT_LSP_COMMANDS = {
    "Python": ["pyright-langserver", "--stdio"],
    "JavaScript": ["typescript-language-server", "--stdio"],
    "JavaScript/JSX": ["typescript-language-server", "--stdio"],
    "TypeScript": ["typescript-language-server", "--stdio"],
    "TypeScript/TSX": ["typescript-language-server", "--stdio"],
    "C": ["clangd", "--background-index=false"],
    "C++": ["clangd", "--background-index=false"],
    "C/C++ Header": ["clangd", "--background-index=false"],
    "C++ Header": ["clangd", "--background-index=false"],
    "Go": ["gopls", "serve"],
    "Rust": ["rust-analyzer"],
    "Java": ["jdtls"],
    "C#": ["omnisharp", "-lsp"],
}

LANGUAGE_SYSTEM_PROMPTS = {
    "Python": "Use Python 3 conventions, preserve imports/types/decorators, respect async/sync semantics, and keep public APIs stable unless the task explicitly changes them.",
    "C": "Use ISO C conventions appropriate to the existing toolchain. Preserve header/source contracts, ownership/lifetime rules, macros, ABI-sensitive signatures, and include ordering.",
    "C++": "Use the project's detected C++ standard and idioms. Preserve RAII, ownership, const-correctness, templates, namespaces, header/source contracts, and ABI-sensitive interfaces.",
    "C/C++ Header": "Treat this as a public interface/header. Preserve declarations, include guards/pragma once, ABI/API contracts, templates/macros, and header/source consistency.",
    "C++ Header": "Treat this as a public C++ interface/header. Preserve declarations, include guards/pragma once, ABI/API contracts, templates/macros, and header/source consistency.",
    "JavaScript": "Preserve the project's module system, runtime version, async behavior, exports/imports, and established JavaScript patterns.",
    "JavaScript/JSX": "Preserve JSX structure, component contracts, hooks/state semantics, module conventions, and the framework/runtime detected in the manifest.",
    "TypeScript": "Preserve TypeScript types, generics, module boundaries, compiler assumptions, and strictness settings. Avoid unnecessary any/unsafe casts.",
    "TypeScript/TSX": "Preserve TSX component contracts, props/types, hooks/state semantics, module conventions, and compiler assumptions.",
    "Go": "Preserve Go package boundaries, interfaces, error handling, goroutine/channel behavior, formatting conventions, and module compatibility.",
    "Rust": "Preserve ownership/borrowing, lifetimes, traits, error propagation, feature flags, and crate/module boundaries.",
    "Java": "Preserve Java package structure, type safety, checked exceptions, annotations, framework conventions, and public method contracts.",
    "Kotlin": "Preserve Kotlin null-safety, coroutines, data/value semantics, visibility, and framework conventions.",
    "C#": "Preserve .NET type/nullability conventions, async/await semantics, dependency injection patterns, project targets, and public contracts.",
    "F#": "Preserve functional composition, immutability, discriminated unions, modules, and .NET interop assumptions.",
    "PHP": "Preserve PHP version compatibility, namespaces, autoloading, framework conventions, and request/runtime semantics.",
    "Ruby": "Preserve Ruby idioms, dynamic dispatch, module/class contracts, and framework conventions.",
    "Swift": "Preserve Swift value/reference semantics, optionals, concurrency, protocol contracts, and platform conventions.",
    "Dart": "Preserve Dart null-safety, async behavior, isolates where applicable, and Flutter/widget conventions.",
    "Lua": "Preserve Lua table/metatable semantics, module conventions, and runtime compatibility.",
    "Shell": "Preserve shell portability, quoting, exit codes, variable scope, and command safety. Do not introduce interactive assumptions.",
    "SQL": "Preserve SQL dialect and transaction semantics detected from the repository. Never silently change destructive behavior.",
    "HTML": "Preserve semantic structure, accessibility, framework/template bindings, and existing script/style contracts.",
    "CSS": "Preserve cascade, specificity, responsive behavior, selectors, and framework conventions; avoid broad selectors that alter unrelated UI.",
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
    repo_map: str
    retrieval_context: str
    lsp_context: str
    baseline_lsp: dict[str, Any]

    plan_stack: list[dict[str, Any]]
    current_task: dict[str, Any] | None
    issue_report: str

    review_attempts: int
    iterations: int
    file_attempts: dict[str, int]
    task_attempts: dict[str, int]
    task_durations: dict[str, float]
    task_history: list[dict[str, Any]]
    completed_tasks: list[str]
    total_tasks: int

    run_dir: Path
    start_time: float
    elapsed_seconds: float
    editor_status: str
    workflow_status: str
    last_edit: dict[str, Any]
    evaluation: dict[str, Any]
    failure_reason: str
    manifest_refresh_count: int


# ==========================================
# 2. MODEL / LOGGING / JSON UTILITIES
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


def response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def invoke_llm(prompt: str, system_prompt: str | None = None) -> str:
    if system_prompt:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
    else:
        response = llm.invoke(prompt)
    return response_text(response)


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
        starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
        if not starts:
            raise ValueError("LLM output did not contain JSON.")
        start = min(starts)
        for end in range(len(cleaned), start, -1):
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                continue
    raise ValueError("Unable to parse JSON from LLM response.")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text_file(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    if b"\x00" in data:
        raise ValueError(f"Binary file cannot be edited as text: {path}")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8: {path}") from exc
    return text if limit is None else text[:limit]

def is_sensitive_path(path: Path) -> bool:
    name = path.name.lower()
    if name in {value.lower() for value in SENSITIVE_NAMES}:
        return True
    return any(token in name for token in ("secret", "credential", "private_key", "apikey"))


def is_probably_text(path: Path) -> bool:
    if path.name in {"Dockerfile", "Makefile", "CMakeLists.txt", ".gitignore", ".dockerignore", ".editorconfig"}:
        return True
    return path.suffix in TEXT_EXTENSIONS


def relative_project_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    clean = str(relative_path).replace("\\", "/").strip()
    candidate = (root / clean).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Target path escapes project root: {relative_path}") from exc
    return candidate


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.agent_tmp_{os.getpid()}_{time.time_ns()}")
    try:
        temp_path.write_text(content, encoding="utf-8", newline="")
        if path.exists():
            try:
                shutil.copymode(path, temp_path)
            except OSError:
                pass
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

def backup_file(
    project_root: Path,
    run_dir: Path,
    path: Path,
    task_id: str = "task",
    attempt: int = 1,
) -> str | None:
    if not path.exists():
        return None
    rel = relative_project_path(project_root, path)
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:80] or "task"
    backup_dir = run_dir / "backups" / safe_task / f"attempt_{attempt:03d}"
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
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output


# ==========================================
# 3. LANGUAGE / AST / REPOSITORY ANALYSIS
# ==========================================
def infer_language(path: Path, text: str = "") -> str:
    if path.name == "Dockerfile":
        return "Dockerfile"
    if path.name in {"Makefile", "CMakeLists.txt"}:
        return "Build/Config"
    if path.name.endswith(".config"):
        return "Config"

    language = LANGUAGE_BY_EXT.get(path.suffix, "Unknown")
    if path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}:
        sample = text[:16000]
        if any(token in sample for token in (
            "std::", "template<", "template <", "class ", "namespace ",
            "#include <vector>", "#include <string>", "constexpr ", "nullptr",
        )):
            return "C++ Header"
        return "C/C++ Header"
    return language

def tree_sitter_parser(language: str):
    key = TREE_SITTER_LANGUAGE_KEYS.get(language)
    if not key:
        return None
    if TREE_SITTER_AVAILABLE and tslp_get_parser:
        try:
            return tslp_get_parser(key)
        except Exception:
            pass
    if legacy_get_parser:
        try:
            return legacy_get_parser(key)
        except Exception:
            pass
    return None


def node_text(node: Any, source_bytes: bytes) -> str:
    try:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def descendant_identifier(node: Any, source_bytes: bytes) -> str:
    preferred_types = {
        "identifier", "type_identifier", "field_identifier", "property_identifier", "shorthand_property_identifier",
    }
    stack = [node]
    while stack:
        current = stack.pop()
        if getattr(current, "type", "") in preferred_types:
            value = node_text(current, source_bytes).strip()
            if value:
                return value
        try:
            stack.extend(reversed(current.children))
        except Exception:
            continue
    return ""


def tree_sitter_symbols(text: str, language: str) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    parser = tree_sitter_parser(language)
    if parser is None:
        return [], [], []
    source_bytes = text.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return [], [], []

    symbol_kinds = {
        "function_definition": "function", "function_declaration": "function", "function_item": "function",
        "method_definition": "method", "method_declaration": "method", "method_item": "method",
        "class_definition": "class", "class_declaration": "class", "class_specifier": "class",
        "struct_item": "struct", "struct_specifier": "struct", "interface_declaration": "interface",
        "trait_item": "trait", "enum_item": "enum", "enum_declaration": "enum", "type_declaration": "type",
        "impl_item": "impl", "namespace_definition": "namespace", "module": "module",
    }

    symbols: list[dict[str, Any]] = []
    references: list[str] = []
    imports: list[dict[str, Any]] = []
    seen_symbol_keys: set[tuple[str, int]] = set()

    def walk(node: Any, depth: int = 0) -> None:
        node_type = getattr(node, "type", "")
        if node_type in symbol_kinds:
            name = ""
            for field_name in ("name", "declarator", "type"):
                try:
                    child = node.child_by_field_name(field_name)
                except Exception:
                    child = None
                if child is not None:
                    name = node_text(child, source_bytes).strip()
                    if field_name != "name" and not re.fullmatch(r"[A-Za-z_$~][A-Za-z0-9_$~]*", name):
                        name = descendant_identifier(child, source_bytes)
                    if name:
                        break
            if name and re.fullmatch(r"[A-Za-z_$~][A-Za-z0-9_$~]*", name):
                key = (name, int(node.start_point[0]))
                if key not in seen_symbol_keys:
                    seen_symbol_keys.add(key)
                    symbols.append({
                        "name": name,
                        "kind": symbol_kinds[node_type],
                        "start_line": int(node.start_point[0]) + 1,
                        "end_line": int(node.end_point[0]) + 1,
                        "node_type": node_type,
                    })

        if "import" in node_type or "include" in node_type:
            snippet = node_text(node, source_bytes).strip()
            if snippet:
                imports.append({
                    "line": int(node.start_point[0]) + 1,
                    "text": snippet[:500],
                })

        if node_type in {"call", "call_expression", "method_invocation", "function_call", "macro_invocation", "new_expression"}:
            name = descendant_identifier(node, source_bytes)
            if name:
                references.append(name)

        try:
            for child in node.children:
                walk(child, depth + 1)
        except Exception:
            return

    walk(tree.root_node)
    parse_errors = []
    stack = [tree.root_node]
    while stack:
        current = stack.pop()
        if getattr(current, "is_error", False) or getattr(current, "is_missing", False):
            parse_errors.append({
                "line": int(current.start_point[0]) + 1,
                "column": int(current.start_point[1]) + 1,
                "node_type": getattr(current, "type", "error"),
            })
        try:
            stack.extend(current.children)
        except Exception:
            continue

    return symbols[:400], sorted(set(references))[:1000], imports[:250]


def fallback_symbols(text: str, language: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    symbols: list[dict[str, Any]] = []

    def add(name: str, kind: str, line: int, end_line: int | None = None) -> None:
        if name:
            symbols.append({"name": name, "kind": kind, "start_line": line, "end_line": end_line or line})

    if language == "Python":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(node.name, "function", node.lineno, getattr(node, "end_lineno", node.lineno))
                elif isinstance(node, ast.ClassDef):
                    add(node.name, "class", node.lineno, getattr(node, "end_lineno", node.lineno))
            return symbols[:400]
        except SyntaxError:
            pass

    patterns: list[tuple[str, str]] = []
    if language in {"JavaScript", "JavaScript/JSX", "TypeScript", "TypeScript/TSX"}:
        patterns = [
            (r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", "function"),
            (r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", "function"),
            (r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", "class"),
            (r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)", "interface"),
        ]
    elif language in {"Java", "Kotlin", "Scala", "C#", "F#", "PHP"}:
        patterns = [
            (r"^\s*(?:public|private|protected|internal|static|final|abstract|suspend|async|override|virtual|sealed|inline|open|operator|extern|unsafe|partial|new|readonly|\s)*\b(?:[\w<>\[\],?.:]+)\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*(?:public|private|protected|internal|static|abstract|final|sealed|open)?\s*(?:class|interface|record|struct)\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Go":
        patterns = [
            (r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*type\s+([A-Za-z_]\w*)\s+struct", "struct"),
            (r"^\s*type\s+([A-Za-z_]\w*)\s+interface", "interface"),
        ]
    elif language == "Rust":
        patterns = [
            (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)", "struct"),
            (r"^\s*(?:pub\s+)?(?:trait|enum|mod)\s+([A-Za-z_]\w*)", "type"),
        ]
    elif language in {"C", "C/C++ Header", "C++", "C++ Header", "Objective-C", "Objective-C++"}:
        patterns = [
            (r"^\s*(?:[\w:*&<>,~\[\]\\\s]+)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?", "function"),
            (r"^\s*(?:class|struct|union|enum)\s+([A-Za-z_]\w*)", "type"),
        ]
    elif language == "Swift":
        patterns = [
            (r"^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|static\s+|mutating\s+|override\s+)*func\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*(?:class|struct|protocol|enum|actor)\s+([A-Za-z_]\w*)", "type"),
        ]
    elif language == "Ruby":
        patterns = [
            (r"^\s*def\s+([A-Za-z_]\w*[!?=]?)", "function"),
            (r"^\s*class\s+([A-Za-z_]\w*)", "class"),
            (r"^\s*module\s+([A-Za-z_]\w*)", "module"),
        ]
    elif language == "Dart":
        patterns = [
            (r"^\s*(?:Future<[^>]+>|[\w<>?]+)\s+([A-Za-z_]\w*)\s*\(", "function"),
            (r"^\s*class\s+([A-Za-z_]\w*)", "class"),
        ]
    elif language == "Lua":
        patterns = [(r"^\s*(?:local\s+)?function\s+([A-Za-z_][\w.]*)\s*\(", "function")]
    elif language == "Shell":
        patterns = [(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*\(\s*\)\s*\{", "function")]

    for line_no, line in enumerate(lines, 1):
        for pattern, kind in patterns:
            match = re.search(pattern, line)
            if match:
                add(match.group(1), kind, line_no)
                break
    return symbols[:400]


def extract_symbols_and_edges(text: str, language: str) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    symbols, references, imports = tree_sitter_symbols(text, language)
    if not symbols:
        symbols = fallback_symbols(text, language)
    if not imports:
        imports = extract_imports_regex(text, language)
    calls = [{"name": name} for name in references]
    return symbols, references, imports, calls


def extract_imports_regex(text: str, language: str) -> list[dict[str, Any]]:
    patterns: list[str]
    if language == "Python":
        patterns = [r"^\s*(?:from\s+[^#\n]+\s+)?import\s+[^#\n]+"]
    elif language in {"JavaScript", "JavaScript/JSX", "TypeScript", "TypeScript/TSX"}:
        patterns = [r"^\s*import\s+[^;]+;?", r"\brequire\(\s*['\"][^'\"]+['\"]\s*\)"]
    elif language in {"C", "C++", "C/C++ Header", "C++ Header", "Objective-C", "Objective-C++"}:
        patterns = [r"^\s*#\s*include\s*[<\"][^>\"]+[>\"]"]
    elif language == "Go":
        patterns = [r"^\s*import\s+(?:\(|\"[^\"]+\")"]
    elif language == "Rust":
        patterns = [r"^\s*(?:use|mod)\s+[^;]+;?"]
    elif language == "Java":
        patterns = [r"^\s*(?:import|package)\s+[^;]+;"]
    elif language in {"Kotlin", "Scala"}:
        patterns = [r"^\s*(?:import|package)\s+[^\n]+"]
    elif language == "C#":
        patterns = [r"^\s*using\s+[^;]+;"]
    else:
        patterns = []
    results = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for pattern in patterns:
            if re.search(pattern, line):
                results.append({"line": line_no, "text": line.strip()[:500]})
                break
    return results[:250]


def framework_detection(project_root: Path, file_paths: set[str]) -> list[str]:
    detected: set[str] = set()

    def parse_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(read_text_file(path, MAX_FILE_BYTES))
        except Exception:
            return {}

    package = parse_json(project_root / "package.json")
    deps = {}
    deps.update(package.get("dependencies", {}))
    deps.update(package.get("devDependencies", {}))
    node_map = {
        "react": "React", "next": "Next.js", "vue": "Vue", "@angular/core": "Angular", "svelte": "Svelte",
        "express": "Express", "@nestjs/core": "NestJS", "electron": "Electron", "vite": "Vite",
        "webpack": "Webpack", "tailwindcss": "Tailwind CSS", "react-native": "React Native",
    }
    detected.update(value for key, value in node_map.items() if key in deps)

    python_deps: set[str] = set()
    for filename in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
        path = project_root / filename
        if path.exists():
            try:
                for line in read_text_file(path, MAX_FILE_BYTES).splitlines():
                    value = re.split(r"[<>=!~\[]", line.strip().lower(), 1)[0]
                    if value and not value.startswith("#"):
                        python_deps.add(value)
            except Exception:
                pass
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        try:
            python_deps.update(
                match.lower()
                for match in re.findall(r"['\"]([A-Za-z0-9_.-]+)(?:[<>=!~\[])", read_text_file(pyproject, MAX_FILE_BYTES))
            )
        except Exception:
            pass
    python_map = {
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI", "streamlit": "Streamlit",
        "tensorflow": "TensorFlow", "torch": "PyTorch", "pytorch": "PyTorch", "langchain": "LangChain",
        "langgraph": "LangGraph", "sqlalchemy": "SQLAlchemy", "pydantic": "Pydantic",
    }
    detected.update(value for key, value in python_map.items() if key in python_deps)

    lowered_paths = {path.lower() for path in file_paths}
    if "manage.py" in lowered_paths:
        detected.add("Django")
    if "pubspec.yaml" in lowered_paths:
        detected.add("Flutter")
    if any(path.endswith(".csproj") or path.endswith(".sln") for path in lowered_paths):
        detected.add(".NET")
    if "pom.xml" in lowered_paths or "build.gradle" in lowered_paths or "build.gradle.kts" in lowered_paths:
        try:
            for filename in ("pom.xml", "build.gradle", "build.gradle.kts"):
                path = project_root / filename
                if path.exists():
                    body = read_text_file(path, MAX_FILE_BYTES).lower()
                    if "spring-boot" in body or "org.springframework" in body:
                        detected.add("Spring Boot/Spring")
        except Exception:
            pass
    if any(path.endswith("cargo.toml") for path in lowered_paths):
        detected.add("Cargo")
    if "go.mod" in lowered_paths:
        detected.add("Go Modules")
    if "cmakelists.txt" in lowered_paths:
        detected.add("CMake")
    return sorted(detected)


def architecture_detection(files: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [item["path"] for item in files]
    parts = [Path(path).parts for path in paths]
    top_level_dirs = sorted({p[0] for p in parts if len(p) > 1})
    signals: set[str] = set()
    marker_map = {
        "components": {"components"}, "controllers": {"controllers"}, "services": {"services"},
        "repositories": {"repositories"}, "models": {"models"}, "views": {"views"}, "routes": {"routes"},
        "routers": {"router", "routers"}, "handlers": {"handlers"}, "middleware": {"middleware"},
        "utils": {"utils", "util"}, "tests": {"tests", "test"}, "src-layout": {"src"},
        "include-layout": {"include", "includes"}, "lib-layout": {"lib"},
    }
    for label, markers in marker_map.items():
        if any(any(part.lower() in markers for part in path_parts[:-1]) for path_parts in parts):
            signals.add(label)
    basenames = {Path(path).name.lower() for path in paths}
    if basenames & {"main.py", "app.py", "main.go", "main.rs", "main.cpp", "main.c", "main.cc"}:
        signals.add("entrypoint")
    if any(path.lower().endswith(".csproj") for path in paths):
        signals.add("dotnet-project")
    if "manage.py" in basenames:
        signals.add("django-conventions")
    if basenames & {"package.json", "pnpm-workspace.yaml", "yarn.lock", "package-lock.json"}:
        signals.add("node-project")

    if {"controllers", "services", "repositories"} <= signals:
        style = "layered"
    elif "components" in signals:
        style = "component-oriented"
    elif {"models", "views"} <= signals:
        style = "MVC-like"
    elif "include-layout" in signals and "src-layout" in signals:
        style = "compiled-language source/header layout"
    elif "django-conventions" in signals:
        style = "Django-conventional"
    elif "entrypoint" in signals:
        style = "single-service-or-script"
    else:
        style = "repository-conventional"
    return {"style": style, "signals": sorted(signals), "top_level_directories": top_level_dirs[:100]}

def lexical_similarity(query: str, text: str) -> float:
    q = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.$-]+", query.lower()))
    t = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.$-]+", text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def repository_tree(files: list[dict[str, Any]], limit: int = 240) -> str:
    return "\n".join(item["path"] for item in files[:limit])


def collect_repository_context(project_root: Path, exclude_root: Path | None = None) -> dict[str, Any]:
    project_root = project_root.resolve()
    excluded = exclude_root.resolve() if exclude_root else None
    files: list[dict[str, Any]] = []
    language_counts: dict[str, int] = {}
    all_paths: set[str] = set()
    total_bytes = 0
    truncated_scan = False

    for root, dirs, filenames in os.walk(project_root, followlinks=False):
        root_path = Path(root).resolve()
        if excluded and (root_path == excluded or excluded in root_path.parents):
            dirs[:] = []
            continue
        kept_dirs = []
        for directory in dirs:
            candidate = Path(root) / directory
            if directory in IGNORED_DIRS or is_sensitive_path(candidate) or candidate.is_symlink():
                continue
            try:
                resolved = candidate.resolve()
                resolved.relative_to(project_root)
            except (OSError, ValueError):
                continue
            if excluded and (resolved == excluded or excluded in resolved.parents):
                continue
            kept_dirs.append(directory)
        dirs[:] = sorted(kept_dirs)

        for filename in sorted(filenames):
            if len(files) >= MAX_FILES_TO_SCAN:
                truncated_scan = True
                break
            path = Path(root) / filename
            if path.is_symlink() or is_sensitive_path(path) or not is_probably_text(path):
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(project_root)
                if excluded and (resolved == excluded or excluded in resolved.parents):
                    continue
                size = resolved.stat().st_size
            except (OSError, ValueError):
                continue
            if size > MAX_FILE_BYTES:
                continue
            try:
                raw = resolved.read_bytes()
                if b"\x00" in raw:
                    continue
                text_content = raw.decode("utf-8-sig")
            except (OSError, UnicodeDecodeError):
                continue

            relative = resolved.relative_to(project_root).as_posix()
            language = infer_language(resolved, text_content)
            symbols, references, imports, calls = extract_symbols_and_edges(text_content, language)
            item = {
                "path": relative,
                "language": language,
                "size_bytes": size,
                "sha256": sha256_bytes(raw),
                "symbols": symbols,
                "references": references,
                "imports": imports,
                "calls": calls,
                "parse_engine": "tree-sitter" if tree_sitter_parser(language) is not None else "fallback",
            }
            files.append(item)
            all_paths.add(relative)
            language_counts[language] = language_counts.get(language, 0) + 1
            total_bytes += size
        if truncated_scan:
            break

    files.sort(key=lambda item: item["path"])
    symbol_index: dict[str, list[str]] = {}
    for item in files:
        for symbol in item["symbols"]:
            symbol_index.setdefault(symbol["name"], []).append(item["path"])

    file_graph: dict[str, dict[str, float]] = {item["path"]: {} for item in files}
    file_set = {item["path"] for item in files}
    for item in files:
        source_file = item["path"]
        for name in set(item.get("references", [])):
            targets = symbol_index.get(name, [])
            if len(targets) > 25:
                # Ambiguous names should not create a near-complete graph.
                targets = sorted(targets, key=lambda p: (-float(manifest_rank_hint(files, p)), p))[:25]
            for target in targets:
                if target != source_file:
                    file_graph[source_file][target] = file_graph[source_file].get(target, 0.0) + 1.0
        for import_item in item.get("imports", []):
            for target in resolve_import_targets(project_root, source_file, import_item.get("text", ""), file_set):
                if target != source_file:
                    file_graph[source_file][target] = file_graph[source_file].get(target, 0.0) + 2.0

    page_rank = compute_pagerank(file_graph)
    languages = [
        {"name": name, "file_count": count}
        for name, count in sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    entrypoint_names = {
        "main.py", "app.py", "manage.py", "main.go", "main.rs", "main.c", "main.cpp", "index.js", "index.ts",
        "server.js", "server.ts", "main.java", "application.java", "program.cs", "main.dart",
    }
    entrypoints = [item["path"] for item in files if Path(item["path"]).name.lower() in entrypoint_names]

    ranked_symbols = build_repo_map(files, page_rank, max_symbols=MAX_REPO_MAP_SYMBOLS)
    manifest = {
        "schema_version": "5.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": {
            "address": str(project_root),
            "name": project_root.name,
            "git_repository": (project_root / ".git").exists(),
        },
        "languages": languages,
        "frameworks": framework_detection(project_root, all_paths),
        "architecture": architecture_detection(files),
        "entrypoints": entrypoints,
        "files": files,
        "file_graph": file_graph,
        "page_rank": page_rank,
        "repo_map": ranked_symbols,
        "directory_tree": repository_tree(files),
        "analysis_capabilities": {
            "tree_sitter": TREE_SITTER_AVAILABLE,
            "lsp": ENABLE_LSP,
            "rag": ENABLE_RAG and OLLAMA_EMBEDDINGS_AVAILABLE,
        },
        "scan_limits": {
            "max_files": MAX_FILES_TO_SCAN,
            "max_file_bytes": MAX_FILE_BYTES,
            "max_llm_file_bytes": MAX_LLM_FILE_BYTES,
        },
        "scan_truncated": truncated_scan,
        "total_files_scanned": len(files),
        "total_text_bytes_scanned": total_bytes,
    }
    return manifest


def manifest_rank(files: list[dict[str, Any]], path: str) -> float:
    # Cheap deterministic hint used only to cap highly ambiguous symbol edges during scanning.
    item = next((entry for entry in files if entry.get("path") == path), None)
    if not item:
        return 0.0
    symbols = len(item.get("symbols", []))
    return 1.0 / max(1, symbols)

def resolve_import_targets(project_root: Path, source_file: str, import_text: str, file_set: set[str]) -> list[str]:
    raw = import_text.strip()
    candidates: set[str] = set()
    source_path = Path(source_file)
    source_dir = source_path.parent

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", raw)
    if quoted:
        for value in quoted:
            if value.startswith(".") or value.startswith("/"):
                base = (source_dir / value).as_posix().lstrip("./")
                candidates.update(make_import_candidates(base))
    if raw.startswith("#include"):
        quoted_include = re.search(r"[<\"]([^>\"]+)[>\"]", raw)
        if quoted_include:
            candidates.update(make_import_candidates(quoted_include.group(1)))
    if raw.startswith("from ") or raw.startswith("import "):
        py_match = re.search(r"(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)", raw)
        if py_match:
            dotted = py_match.group(1).replace(".", "/")
            candidates.update(make_import_candidates(dotted))
    if raw.startswith("use ") or raw.startswith("mod "):
        rust_match = re.search(r"(?:use|mod)\s+([A-Za-z_][A-Za-z0-9_:]*)", raw)
        if rust_match:
            rust_path = rust_match.group(1).split("::")[0]
            candidates.update(make_import_candidates(rust_path))

    return sorted(candidate for candidate in candidates if candidate in file_set)[:12]


def make_import_candidates(base: str) -> list[str]:
    clean = base.replace("\\", "/").lstrip("./")
    if clean.startswith("src/") or clean.startswith("lib/"):
        root_variants = [clean]
    else:
        root_variants = [clean, f"src/{clean}", f"lib/{clean}"]
    extensions = [
        "", ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".c", ".h", ".cpp", ".hpp", ".cc", ".go", ".rs",
        "/__init__.py", "/index.js", "/index.ts", "/mod.rs",
    ]
    return [f"{root}{ext}" for root in root_variants for ext in extensions]


def compute_pagerank(graph: dict[str, dict[str, float]], damping: float = 0.85, iterations: int = 30) -> dict[str, float]:
    nodes = list(graph)
    if not nodes:
        return {}
    rank = {node: 1.0 / len(nodes) for node in nodes}
    for _ in range(iterations):
        next_rank = {node: (1.0 - damping) / len(nodes) for node in nodes}
        dangling = sum(rank[node] for node in nodes if not graph.get(node))
        dangling_share = damping * dangling / len(nodes)
        for node in nodes:
            next_rank[node] += dangling_share
        for source, edges in graph.items():
            total_weight = sum(edges.values())
            if total_weight <= 0:
                continue
            for target, weight in edges.items():
                next_rank[target] += damping * rank[source] * weight / total_weight
        rank = next_rank
    maximum = max(rank.values()) or 1.0
    return {node: value / maximum for node, value in rank.items()}


def build_repo_map(files: list[dict[str, Any]], page_rank: dict[str, float], max_symbols: int = 240) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    for file_item in files:
        file_rank = float(page_rank.get(file_item["path"], 0.0))
        for symbol in file_item.get("symbols", []):
            symbols.append({
                "file": file_item["path"],
                "language": file_item["language"],
                "name": symbol["name"],
                "kind": symbol["kind"],
                "start_line": symbol.get("start_line", 0),
                "end_line": symbol.get("end_line", 0),
                "pagerank": round(file_rank, 6),
            })
    symbols.sort(key=lambda item: (-item["pagerank"], item["file"], item["start_line"], item["name"]))
    return symbols[:max_symbols]


def rank_files_for_query(manifest: dict[str, Any], query: str, limit: int = 12) -> list[dict[str, Any]]:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.$:-]+", query.lower()))
    scored = []
    for file_item in manifest.get("files", []):
        path = file_item.get("path", "")
        symbol_names = " ".join(symbol.get("name", "") for symbol in file_item.get("symbols", []))
        haystack = f"{path} {file_item.get('language', '')} {symbol_names}"
        lexical = lexical_similarity(query, haystack)
        path_hit = 1.0 if any(token and token in path.lower() for token in tokens) else 0.0
        symbol_hit = 1.0 if any(token and token in symbol_names.lower().split() for token in tokens) else 0.0
        pagerank = float(manifest.get("page_rank", {}).get(path, 0.0))
        score = 0.55 * lexical + 0.20 * path_hit + 0.10 * symbol_hit + 0.15 * pagerank
        if score > 0:
            scored.append({
                "file": path,
                "language": file_item.get("language", "Unknown"),
                "score": round(score, 6),
                "pagerank": round(pagerank, 6),
                "symbols": [symbol.get("name", "") for symbol in file_item.get("symbols", [])[:25]],
            })
    scored.sort(key=lambda item: (-item["score"], -item["pagerank"], item["file"]))
    return scored[:limit]

def manifest_target(manifest: dict[str, Any], target_file: str) -> dict[str, Any]:
    return next((item for item in manifest.get("files", []) if item.get("path") == target_file), {})


def symbol_range(manifest: dict[str, Any], target_file: str, target_symbol: str) -> tuple[int, int] | None:
    item = manifest_target(manifest, target_file)
    target = target_symbol.strip()
    for symbol in item.get("symbols", []):
        if symbol.get("name") == target:
            return int(symbol.get("start_line", 1)), int(symbol.get("end_line", symbol.get("start_line", 1)))
    return None


def language_context(manifest: dict[str, Any], target_file: str) -> str:
    target = manifest_target(manifest, target_file)
    language = target.get("language", "Unknown")
    frameworks = ", ".join(manifest.get("frameworks", [])) or "None detected"
    architecture = manifest.get("architecture", {})
    guidance = LANGUAGE_SYSTEM_PROMPTS.get(language, "Use the repository's existing syntax, conventions, build tooling, and public API contracts.")
    return "\n".join([
        f"Target language: {language}",
        f"Language-specific editor guidance: {guidance}",
        f"Project frameworks/libraries: {frameworks}",
        f"Architecture: {architecture.get('style', 'unknown')}",
        f"Architecture signals: {', '.join(architecture.get('signals', [])) or 'none'}",
    ])


# ==========================================
# 4. LOCAL EMBEDDING INDEX (RAG)
# ==========================================
def ast_aware_chunks(project_root: Path, manifest: dict[str, Any], max_chunks: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for file_item in manifest.get("files", []):
        if len(chunks) >= max_chunks:
            break
        path = resolve_repo_path(project_root, file_item["path"])
        if not path.exists() or is_sensitive_path(path) or path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            text_content = read_text_file(path)
        except Exception:
            continue
        file_hash = str(file_item.get("sha256") or sha256_bytes(path.read_bytes()))
        lines = text_content.splitlines()
        symbols = file_item.get("symbols", [])
        if symbols:
            for symbol in symbols:
                if len(chunks) >= max_chunks:
                    break
                start = max(1, int(symbol.get("start_line", 1)) - 4)
                end = min(len(lines), int(symbol.get("end_line", start)) + 4)
                while end - start + 1 > RAG_CHUNK_LINES:
                    end = max(start, end - 10)
                body = "\n".join(lines[start - 1:end])
                if not body.strip():
                    continue
                chunk_id = hashlib.sha1(f"{file_item['path']}:{file_hash}:{start}:{end}".encode()).hexdigest()[:20]
                chunks.append({
                    "id": chunk_id,
                    "content_sha256": file_hash,
                    "path": file_item["path"],
                    "language": file_item["language"],
                    "symbol": symbol.get("name", ""),
                    "start_line": start,
                    "end_line": end,
                    "text": f"FILE: {file_item['path']}\nSYMBOL: {symbol.get('name')}\n{body}",
                })
        if not symbols and lines:
            for start in range(1, len(lines) + 1, RAG_CHUNK_LINES):
                if len(chunks) >= max_chunks:
                    break
                end = min(len(lines), start + RAG_CHUNK_LINES - 1)
                body = "\n".join(lines[start - 1:end])
                chunk_id = hashlib.sha1(f"{file_item['path']}:{file_hash}:{start}:{end}".encode()).hexdigest()[:20]
                chunks.append({
                    "id": chunk_id,
                    "content_sha256": file_hash,
                    "path": file_item["path"],
                    "language": file_item["language"],
                    "symbol": "",
                    "start_line": start,
                    "end_line": end,
                    "text": f"FILE: {file_item['path']}\n{body}",
                })
    return chunks

class LocalEmbeddingIndex:
    def __init__(self, path: Path, embeddings: Any):
        self.path = path
        self.embeddings = embeddings
        self.records: list[dict[str, Any]] = []

    @classmethod
    def build_or_load(cls, path: Path, chunks: list[dict[str, Any]]) -> "LocalEmbeddingIndex | None":
        if not ENABLE_RAG or not OLLAMA_EMBEDDINGS_AVAILABLE or OllamaEmbeddings is None:
            return None
        try:
            kwargs: dict[str, Any] = {"model": OLLAMA_EMBED_MODEL}
            if OLLAMA_BASE_URL:
                kwargs["base_url"] = OLLAMA_BASE_URL
            embedder = OllamaEmbeddings(**kwargs)
            index = cls(path, embedder)
            wanted = {chunk["id"]: chunk for chunk in chunks}
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("model") == OLLAMA_EMBED_MODEL and isinstance(payload.get("records"), list):
                    existing_records = [record for record in payload["records"] if isinstance(record, dict)]
                    index.records = [
                        record for record in existing_records
                        if record.get("id") in wanted
                        and record.get("content_sha256") == wanted[record.get("id")].get("content_sha256")
                    ]
            existing = {record.get("id") for record in index.records}
            pending = [wanted[cid] for cid in sorted(wanted) if cid not in existing]
            for offset in range(0, len(pending), RAG_BATCH_SIZE):
                batch = pending[offset:offset + RAG_BATCH_SIZE]
                vectors = embedder.embed_documents([item["text"] for item in batch])
                for item, vector in zip(batch, vectors):
                    index.records.append({**item, "vector": vector})
                index._persist()
            index._persist()
            return index
        except Exception as exc:
            raise RuntimeError(f"Local Ollama embedding index unavailable: {exc}") from exc

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "model": OLLAMA_EMBED_MODEL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "records": self.records,
        }
        atomic_write_text(self.path, json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, limit: int = MAX_RETRIEVAL_RESULTS) -> list[dict[str, Any]]:
        if not self.records or not query.strip():
            return []
        vector = self.embeddings.embed_query(query)
        scored = []
        for record in self.records:
            score = self.cosine(vector, record.get("vector", []))
            scored.append({k: v for k, v in record.items() if k != "vector"} | {"score": round(score, 6)})
        scored.sort(key=lambda item: (-item["score"], item.get("path", ""), item.get("start_line", 0)))
        return scored[:limit]


def load_rag_index(run_dir: Path) -> LocalEmbeddingIndex | None:
    if not ENABLE_RAG or not OLLAMA_EMBEDDINGS_AVAILABLE or OllamaEmbeddings is None:
        return None
    path = run_dir / "rag_index.json"
    if not path.exists():
        return None
    try:
        kwargs: dict[str, Any] = {"model": OLLAMA_EMBED_MODEL}
        if OLLAMA_BASE_URL:
            kwargs["base_url"] = OLLAMA_BASE_URL
        index = LocalEmbeddingIndex(path, OllamaEmbeddings(**kwargs))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != OLLAMA_EMBED_MODEL:
            return None
        index.records = payload.get("records", [])
        return index
    except Exception:
        return None


# ==========================================
# 5. LSP INTEGRATION
# ==========================================
def configured_lsp_command(language: str) -> list[str] | None:
    env_name = "LSP_COMMAND_" + re.sub(r"[^A-Za-z0-9]", "_", language).upper()
    configured = os.getenv(env_name)
    if configured:
        return shlex.split(configured)
    command = DEFAULT_LSP_COMMANDS.get(language)
    if not command:
        return None
    executable = shutil.which(command[0])
    if executable is None:
        return None
    return command


class LSPClient:
    def __init__(self, command: list[str], project_root: Path):
        self.command = command
        self.project_root = project_root.resolve()
        self.process: subprocess.Popen[bytes] | None = None
        self._next_id = 1
        self._opened_uri: str | None = None
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = threading.Event()
        self._notifications: list[dict[str, Any]] = []
        self._pending_by_id: dict[int, dict[str, Any]] = {}

    def start(self) -> None:
        if self.process is not None:
            return
        self.process = subprocess.Popen(
            self.command,
            cwd=self.project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
            bufsize=0,
        )
        self._stop_reader.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="lsp-reader", daemon=True)
        self._reader_thread.start()
        root_uri = self.project_root.as_uri()
        init = self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "rootPath": str(self.project_root),
            "workspaceFolders": [{"uri": root_uri, "name": self.project_root.name}],
            "capabilities": {
                "workspace": {"workspaceFolders": True},
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "references": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                },
            },
            "clientInfo": {"name": "local-langgraph-editor", "version": "2.0"},
        }, timeout=LSP_TIMEOUT_SECONDS)
        if init and init.get("error"):
            raise RuntimeError(f"LSP initialize failed: {init['error']}")
        self.notify("initialized", {})

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            shutdown = self.request("shutdown", None, timeout=min(2.0, LSP_TIMEOUT_SECONDS))
            if shutdown and shutdown.get("error"):
                pass
            self.notify("exit", None)
        except Exception:
            pass
        self._stop_reader.set()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        try:
            process.kill()
            process.wait(timeout=2)
        except Exception:
            pass
        self.process = None
        self._opened_uri = None
        self._reader_thread = None
        self._pending_by_id.clear()

    def _write_message(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("LSP process is not running")
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + payload)
        self.process.stdin.flush()

    def notify(self, method: str, params: Any) -> None:
        self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout: float = LSP_TIMEOUT_SECONDS) -> dict[str, Any] | None:
        request_id = self._next_id
        self._next_id += 1
        self._write_message({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        if request_id in self._pending_by_id:
            result = self._pending_by_id.pop(request_id)
            return result
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                message = self._queue.get(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(f"LSP server exited with code {self.process.returncode}")
                continue
            if message.get("method") == "textDocument/publishDiagnostics":
                self._notifications.append(message)
                continue
            msg_id = message.get("id")
            if msg_id == request_id:
                return message
            if isinstance(msg_id, int):
                self._pending_by_id[msg_id] = message
        return {"error": {"code": -32002, "message": f"LSP request timed out: {method}"}}

    def _reader_loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        stream = self.process.stdout
        while not self._stop_reader.is_set():
            try:
                headers: dict[str, str] = {}
                while True:
                    line = stream.readline()
                    if not line:
                        return
                    decoded = line.decode("ascii", errors="replace").strip()
                    if not decoded:
                        break
                    key, sep, value = decoded.partition(":")
                    if sep:
                        headers[key.lower()] = value.strip()
                length = int(headers.get("content-length", "0"))
                if length <= 0:
                    continue
                body = stream.read(length)
                if not body:
                    return
                message = json.loads(body.decode("utf-8"))
                self._queue.put(message)
            except (OSError, ValueError, json.JSONDecodeError):
                return

    def open_document(self, path: Path, language: str, text: str) -> str:
        uri = path.resolve().as_uri()
        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": LANGUAGE_LSP_IDS.get(language, language.lower()),
                "version": 1,
                "text": text,
            }
        })
        self._opened_uri = uri
        return uri

    def close_document(self) -> None:
        if self._opened_uri:
            self.notify("textDocument/didClose", {"textDocument": {"uri": self._opened_uri}})
            self._opened_uri = None

    def analyze_document(self, uri: str, line: int = 0, character: int = 0) -> dict[str, Any]:
        position = {"line": max(0, line), "character": max(0, character)}
        result = self.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        definition = self.request("textDocument/definition", {"textDocument": {"uri": uri}, "position": position})
        references = self.request("textDocument/references", {
            "textDocument": {"uri": uri}, "position": position, "context": {"includeDeclaration": True}
        })
        diagnostics = [
            item.get("params", {})
            for item in self._notifications
            if item.get("method") == "textDocument/publishDiagnostics"
            and item.get("params", {}).get("uri") == uri
        ]
        return {
            "document_symbols": result.get("result") if result and "error" not in result else None,
            "definition": definition.get("result") if definition and "error" not in definition else None,
            "references": references.get("result") if references and "error" not in references else None,
            "diagnostics": [diag for group in diagnostics for diag in group.get("diagnostics", [])],
        }


def lsp_analyze_file(project_root: Path, target_file: str, language: str, text: str, line: int = 0, character: int = 0) -> dict[str, Any]:
    if not ENABLE_LSP:
        return {"available": False, "reason": "disabled"}
    command = configured_lsp_command(language)
    if not command:
        return {"available": False, "reason": f"No installed LSP server configured for {language}."}
    client = LSPClient(command, project_root)
    try:
        client.start()
        target = resolve_repo_path(project_root, target_file)
        uri = client.open_document(target, language, text)
        time.sleep(0.20)
        result = client.analyze_document(uri, line=line, character=character)
        result["available"] = True
        result["command"] = command
        return result
    except Exception as exc:
        return {"available": False, "reason": f"LSP failure: {exc}", "command": command}
    finally:
        try:
            client.close_document()
        except Exception:
            pass
        client.stop()

def render_lsp_context(result: dict[str, Any]) -> str:
    if not result.get("available"):
        return f"LSP unavailable: {result.get('reason', 'unknown reason')}"
    lines = [f"LSP command: {' '.join(result.get('command', []))}"]
    diagnostics = result.get("diagnostics", [])
    if diagnostics:
        lines.append("Diagnostics:")
        lines.extend(json.dumps(item, ensure_ascii=False) for item in diagnostics[:25])
    else:
        lines.append("Diagnostics: none reported during the analysis window.")
    definition = result.get("definition")
    if definition:
        lines.append(f"Definition: {json.dumps(definition, ensure_ascii=False)[:6000]}")
    refs = result.get("references")
    if refs:
        lines.append(f"References: {json.dumps(refs[:30], ensure_ascii=False)[:12000]}")
    return "\n".join(lines)


def compact_manifest_for_llm(manifest: dict[str, Any], max_chars: int = MAX_MANIFEST_PROMPT_CHARS) -> str:
    compact = {
        "schema_version": manifest.get("schema_version"),
        "project": manifest.get("project"),
        "languages": manifest.get("languages", [])[:12],
        "frameworks": manifest.get("frameworks", [])[:20],
        "architecture": manifest.get("architecture", {}),
        "entrypoints": manifest.get("entrypoints", [])[:30],
        "files": [
            {
                "path": item.get("path"),
                "language": item.get("language"),
                "size_bytes": item.get("size_bytes"),
                "symbols": item.get("symbols", [])[:30],
            }
            for item in manifest.get("files", [])
        ],
        "repo_map": manifest.get("repo_map", [])[:120],
    }
    raw = json.dumps(compact, indent=2, ensure_ascii=False)
    return raw[:max_chars]


def related_files_from_graph(manifest: dict[str, Any], seed_files: list[str], limit: int = 8) -> list[str]:
    graph = manifest.get("file_graph", {})
    reverse: dict[str, set[str]] = {}
    for source, edges in graph.items():
        for target in edges:
            reverse.setdefault(target, set()).add(source)
    scores: dict[str, float] = {}
    for seed in seed_files:
        for target, weight in graph.get(seed, {}).items():
            scores[target] = scores.get(target, 0.0) + 2.0 * float(weight)
        for source in reverse.get(seed, set()):
            scores[source] = scores.get(source, 0.0) + 1.5
    for path, rank in manifest.get("page_rank", {}).items():
        if path not in seed_files:
            scores[path] = scores.get(path, 0.0) + 0.15 * float(rank)
    return [path for path, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]

# ==========================================
# 6. LAYERED CONTEXT LOADING
# ==========================================
def build_layered_context(
    project_root: Path,
    manifest: dict[str, Any],
    query: str,
    target_file: str | None = None,
    target_symbol: str | None = None,
    include_full_target: bool = False,
) -> dict[str, Any]:
    overview = "\n".join([
        f"Project: {manifest.get('project', {}).get('name', project_root.name)}",
        f"Languages: {', '.join(item['name'] for item in manifest.get('languages', [])[:12]) or 'unknown'}",
        f"Frameworks: {', '.join(manifest.get('frameworks', [])) or 'none detected'}",
        f"Architecture: {manifest.get('architecture', {}).get('style', 'unknown')}",
        "Directory overview:",
        manifest.get("directory_tree", "")[:7000],
    ])
    ranked = rank_files_for_query(manifest, query, limit=10)
    seed_files = [item["file"] for item in ranked[:4]]
    graph_related = related_files_from_graph(manifest, seed_files, limit=8)
    map_lines = [
        f"{item['file']}::{item['name']} [{item['kind']}] lines {item['start_line']}-{item['end_line']} rank={item['pagerank']}"
        for item in manifest.get("repo_map", [])[:120]
    ]
    context: dict[str, Any] = {
        "overview": overview,
        "ranked_files": ranked,
        "repo_map": "\n".join(map_lines),
        "rag_results": [],
        "target": "",
        "related": "",
    }
    if target_file:
        target_path = resolve_repo_path(project_root, target_file)
        target_manifest = manifest_target(manifest, target_file)
        if target_path.exists() and target_path.is_file():
            source = read_text_file(target_path)
            target_context = source if include_full_target and len(source.encode("utf-8")) <= MAX_LLM_FILE_BYTES else extract_target_context(
                source, target_manifest, target_symbol or ""
            )
            context["target"] = f"TARGET FILE: {target_file}\n{target_context}"
            related_paths = [p for p in graph_related if p != target_file][:4]
            related_parts = []
            for path in related_paths:
                try:
                    related_manifest = manifest_target(manifest, path)
                    related_source = read_text_file(resolve_repo_path(project_root, path), min(MAX_LLM_FILE_BYTES, 10000))
                    snippets = extract_relevant_symbol_snippets(related_source, related_manifest, limit_symbols=3)
                    if snippets:
                        related_parts.append(f"--- {path} ---\n{snippets}")
                except Exception:
                    continue
            context["related"] = "\n".join(related_parts)
    return context

def build_layered_context_with_rag(
    project_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    query: str,
    target_file: str | None = None,
    target_symbol: str | None = None,
    include_full_target: bool = False,
) -> dict[str, Any]:
    context = build_layered_context(
        project_root, manifest, query, target_file=target_file, target_symbol=target_symbol, include_full_target=include_full_target
    )
    index = load_rag_index(run_dir)
    if index is not None:
        try:
            context["rag_results"] = index.search(query, limit=MAX_RETRIEVAL_RESULTS)
        except Exception:
            context["rag_results"] = []
    return context


def extract_target_context(source: str, target_manifest: dict[str, Any], target_symbol: str) -> str:
    lines = source.splitlines()
    if not lines:
        return source
    selected_ranges: list[tuple[int, int]] = []
    if target_symbol:
        for symbol in target_manifest.get("symbols", []):
            if symbol.get("name") == target_symbol:
                selected_ranges.append((max(1, symbol.get("start_line", 1) - 10), min(len(lines), symbol.get("end_line", 1) + 12)))
                break
    if not selected_ranges and target_manifest.get("symbols"):
        first = target_manifest["symbols"][0]
        selected_ranges.append((1, min(len(lines), first.get("end_line", 80) + 12)))
    if not selected_ranges:
        selected_ranges.append((1, min(len(lines), 180)))
    # Always expose module header/imports so edits can preserve dependencies.
    header_end = min(len(lines), 100)
    selected_ranges.append((1, header_end))
    return merge_line_ranges(lines, selected_ranges, max_chars=MAX_LLM_FILE_BYTES)


def extract_relevant_symbol_snippets(source: str, file_manifest: dict[str, Any], limit_symbols: int = 4) -> str:
    lines = source.splitlines()
    ranges = []
    for symbol in file_manifest.get("symbols", [])[:limit_symbols]:
        ranges.append((max(1, symbol.get("start_line", 1) - 2), min(len(lines), symbol.get("end_line", 1) + 3)))
    return merge_line_ranges(lines, ranges, max_chars=18000) if ranges else "\n".join(lines[:100])


def merge_line_ranges(lines: list[str], ranges: list[tuple[int, int]], max_chars: int = 30000) -> str:
    if not ranges:
        return ""
    normalized = sorted(ranges)
    merged: list[list[int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    pieces = []
    for start, end in merged:
        pieces.append("\n".join(f"{index}: {lines[index - 1]}" for index in range(start, end + 1)))
    text = "\n\n".join(pieces)
    return text[:max_chars]


def task_source_context(source: str, task: dict[str, Any], manifest_target_data: dict[str, Any]) -> str:
    symbol = str(task.get("function", "")).strip()
    if symbol:
        context = extract_target_context(source, manifest_target_data, symbol)
        if context.strip():
            return context
    start_line = int(task.get("start_line", 0) or 0)
    end_line = int(task.get("end_line", 0) or 0)
    lines = source.splitlines()
    if start_line > 0 and lines:
        start = max(1, start_line - 8)
        end = min(len(lines), max(start, end_line or start_line) + 8)
        return "\n".join(f"{i:5d}: {lines[i-1]}" for i in range(start, end + 1))
    return source[:MAX_LLM_FILE_BYTES]


def reviewer_says_pass(raw: str) -> bool:
    cleaned = raw.replace("```", "").strip()
    if re.match(r"^PASS(?:\s|[.!:,;\-]|$)", cleaned, re.IGNORECASE):
        return True
    try:
        parsed = safe_json_load(cleaned)
        return isinstance(parsed, dict) and str(parsed.get("status", "")).strip().upper() == "PASS"
    except Exception:
        return False


def render_retrieval_context(context: dict[str, Any], include_target: bool = True) -> str:
    ranked = "\n".join(
        f"{item['file']} score={item['score']} symbols={', '.join(item.get('symbols', [])[:12])}"
        for item in context.get("ranked_files", [])
    )
    rag = "\n".join(
        f"{item.get('path')}:{item.get('start_line')}-{item.get('end_line')} score={item.get('score')}\n{item.get('text', '')[:3000]}"
        for item in context.get("rag_results", [])
    )
    sections = [
        ("[LAYER 1 — OVERVIEW]", context.get("overview", "")[:5000]),
        ("[LAYER 2 — RANKED FILES]", ranked[:5000]),
        ("[LAYER 2 — REPO MAP]", context.get("repo_map", "")[:7000]),
        ("[LAYER 3 — SEMANTIC RETRIEVAL]", rag[:7000]),
    ]
    if include_target:
        sections.append(("[LAYER 3 — TARGET]", context.get("target", "")[:12000]))
    sections.append(("[LAYER 2/3 — RELATED]", context.get("related", "")[:6000]))
    return "\n\n".join(f"{header}\n{body}" for header, body in sections if body).strip()[:MAX_CONTEXT_CHARS]

# ==========================================
# 7. REQUIREMENT NODE
# ==========================================
def requirement_node(state: AgentState) -> AgentState:
    project = Path(state["project_address"]).expanduser().resolve()
    request = state["user_request"].strip()
    if not project.is_dir():
        raise ValueError(f"project_address is not a directory: {project}")
    if not request:
        raise ValueError("user_request must not be empty")

    run_dir = state.get("run_dir") or (Path.cwd() / "SDLC_Runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = f"""
You are a senior software requirements engineer modifying an EXISTING LOCAL REPOSITORY.

PROJECT PATH:
{project}

USER REQUEST:
{request}

Convert the request into concise implementation requirements for an editor agent.
Do not invent unrelated features.
Return ONLY JSON with exactly these keys:
{{
  "project_goal": "one concise sentence",
  "functional_requirements": ["..."],
  "non_functional_requirements": ["..."],
  "constraints": ["..."],
  "acceptance_criteria": ["..."]
}}
"""
    write_log(run_dir, "Requirement Node - PROMPT", prompt)
    requirements = invoke_llm(prompt)
    try:
        requirements_object = safe_json_load(requirements)
        requirements = json.dumps(requirements_object, indent=2, ensure_ascii=False)
    except Exception:
        requirements = requirements.strip()
    write_log(run_dir, "Requirement Node - OUTPUT", requirements)
    atomic_write_text(run_dir / "requirements.txt", requirements)

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
        "completed_tasks": state.get("completed_tasks", []),
        "last_edit": {},
        "evaluation": {},
        "workflow_status": "RUNNING",
        "editor_status": "READY",
        "total_tasks": 0,
    }


# ==========================================
# 8. ANALYZER NODE
# ==========================================
def analyzer_node(state: AgentState) -> AgentState:
    project = Path(state["project_address"]).resolve()
    run_dir = Path(state["run_dir"])
    print("\n🔬 [Analyzer] Scanning repository with AST + RepoMap + PageRank + local RAG...")

    manifest = collect_repository_context(project, exclude_root=run_dir)
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    atomic_write_text(run_dir / "manifest.json", manifest_json)
    write_log(run_dir, "Analyzer Node - MANIFEST", manifest_json[:70000])

    rag_status = {"enabled": False, "model": OLLAMA_EMBED_MODEL, "chunks": 0, "error": None}
    if ENABLE_RAG and OLLAMA_EMBEDDINGS_AVAILABLE:
        try:
            chunks = ast_aware_chunks(project, manifest, MAX_RAG_CHUNKS)
            index = LocalEmbeddingIndex.build_or_load(run_dir / "rag_index.json", chunks)
            rag_status.update({"enabled": index is not None, "chunks": len(index.records) if index else 0})
        except Exception as exc:
            rag_status["error"] = str(exc)
            write_log(run_dir, "Analyzer Node - RAG WARNING", rag_status)
    manifest["rag_status"] = rag_status
    manifest["lsp_servers"] = {
        language: configured_lsp_command(language)
        for language in sorted({item["language"] for item in manifest.get("files", [])})
        if configured_lsp_command(language)
    }
    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
    atomic_write_text(run_dir / "manifest.json", manifest_json)
    repo_map = json.dumps(manifest.get("repo_map", []), indent=2, ensure_ascii=False)

    return {"manifest": manifest, "manifest_json": manifest_json, "repo_map": repo_map}


# ==========================================
# 9. PLANNER NODE
# ==========================================
def planner_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    project = Path(state["project_address"])
    manifest = state["manifest"]
    retrieval = build_layered_context_with_rag(
        project,
        run_dir,
        manifest,
        f"{state['user_request']}\n{state['requirements']}",
    )
    retrieval_context = render_retrieval_context(retrieval, include_target=False)

    prompt = f"""
You are the planning agent for an EXISTING local codebase.

USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements']}

REPOSITORY MANIFEST (ground truth):
{compact_manifest_for_llm(state["manifest"], MAX_MANIFEST_PROMPT_CHARS)}

LAYERED CONTEXT:
{retrieval_context}

Create a granular execution plan as a strict LIFO STACK.

STACK RULE:
- The array is stored bottom-to-top.
- The LAST array element is the FIRST task executed with pop().
- One task must normally touch exactly ONE file.
- Existing-file tasks MUST identify an exact function, class, method, handler, declaration, or precise module-level section.
- Create-file tasks must explicitly use action=create and file=new relative path.
- Do not plan formatting-only work or unrelated cleanup.
- Prefer small surgical tasks.
- Include prerequisite edits before dependent edits when that ordering matters.

Every task MUST have:
- id
- action: modify or create
- file: repository-relative path
- function: exact symbol OR precise section name
- change: exact intended behavior
- rationale
- acceptance_criteria: concrete checks
- related_files

Return ONLY JSON:
{{
  "plan_stack": [
    {{
      "id": "T01",
      "action": "modify",
      "file": "src/example.py",
      "function": "example_function",
      "change": "...",
      "rationale": "...",
      "acceptance_criteria": ["..."],
      "related_files": ["src/other.py"]
    }}
  ]
}}
"""
    write_log(run_dir, "Planner Node - PROMPT", prompt)
    raw = invoke_llm(prompt)
    write_log(run_dir, "Planner Node - RAW OUTPUT", raw[:50000])
    parsed = safe_json_load(raw)
    tasks = parsed.get("plan_stack") if isinstance(parsed, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner did not return a non-empty plan_stack")
    if len(tasks) > MAX_PLANNER_TASKS:
        tasks = tasks[:MAX_PLANNER_TASKS]

    known_files = {item["path"] for item in manifest.get("files", [])}
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_task in enumerate(tasks, 1):
        if not isinstance(raw_task, dict):
            raise ValueError(f"Planner task #{index} is not an object")
        task = deepcopy(raw_task)
        task_id = str(task.get("id") or f"T{index:02d}").strip()
        action = str(task.get("action", "")).strip().lower()
        file_path = str(task.get("file", "")).replace("\\", "/").strip()
        target_symbol = str(task.get("function", task.get("target_symbol", ""))).strip()
        change = str(task.get("change", "")).strip()
        if task_id in seen_ids:
            raise ValueError(f"Duplicate planner task id: {task_id}")
        if action not in {"modify", "create"}:
            raise ValueError(f"Task {task_id}: action must be modify or create")
        if not file_path or Path(file_path).is_absolute() or ".." in Path(file_path).parts:
            raise ValueError(f"Task {task_id}: invalid repository-relative file path")
        if not change:
            raise ValueError(f"Task {task_id}: change is required")
        if action == "modify" and file_path not in known_files:
            raise ValueError(f"Task {task_id}: modify target not found in manifest: {file_path}")
        if action == "modify" and not target_symbol:
            raise ValueError(f"Task {task_id}: exact function/class/method/section is required")
        if action == "create" and not target_symbol:
            target_symbol = "new file/module"
        if action == "modify" and target_symbol.lower() not in {"module-level imports", "module-level constants", "configuration", "configuration block", "route table", "declaration block"}:
            symbols = {str(symbol.get("name")) for symbol in manifest_target(manifest, file_path).get("symbols", [])}
            if target_symbol not in symbols:
                write_log(
                    run_dir,
                    f"Planner Node - SYMBOL WARNING [{task_id}]",
                    f"Symbol '{target_symbol}' was not found in extracted manifest for {file_path}; allowing source-level verification.",
                )
        related_files = [
            Path(str(path).replace("\\", "/")).as_posix()
            for path in list(task.get("related_files", []))[:12]
            if isinstance(path, str) and path and ".." not in Path(path).parts and not Path(path).is_absolute()
        ]
        normalized.append({
            "id": task_id,
            "action": action,
            "file": Path(file_path).as_posix(),
            "function": target_symbol,
            "change": change,
            "rationale": str(task.get("rationale", "")),
            "new_file": action == "create",
            "acceptance_criteria": [str(value) for value in task.get("acceptance_criteria", []) if str(value).strip()],
            "related_files": related_files,
            "start_line": int(task.get("start_line", 0) or 0),
            "end_line": int(task.get("end_line", 0) or 0),
            "review_status": "PENDING",
        })
        seen_ids.add(task_id)

    plan_json = json.dumps(normalized, indent=2, ensure_ascii=False)
    atomic_write_text(run_dir / "plan_stack.json", plan_json)
    write_log(run_dir, "Planner Node - STACK", plan_json)
    return {
        "plan_stack": normalized,
        "current_task": None,
        "issue_report": "",
        "review_attempts": 0,
        "total_tasks": len(normalized),
        "retrieval_context": retrieval_context,
    }


# ==========================================
# 10. EDIT OPERATIONS
# ==========================================
def validate_edit_plan(parsed: Any, task: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Editor output must be a JSON object")
    action = str(parsed.get("action", "")).strip().lower()
    file_path = str(parsed.get("file", "")).replace("\\", "/").strip()
    if action not in {"modify", "create"}:
        raise ValueError("Editor action must be modify or create")
    if file_path != Path(task["file"]).as_posix():
        raise ValueError(f"Editor targeted '{file_path}', task targets '{task['file']}'")
    if not file_path or Path(file_path).is_absolute() or ".." in Path(file_path).parts:
        raise ValueError("Editor response contains invalid repository-relative path")

    if action == "create":
        if task["action"] != "create":
            raise ValueError("Existing-file task cannot use create action")
        content = parsed.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Create response must contain non-empty content")
        return {"action": "create", "file": file_path, "content": content, "operations": []}

    if task["action"] != "modify":
        raise ValueError("Create-file task must use create action")
    operations = parsed.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("Modify response must contain at least one search/replace operation")
    if len(operations) > MAX_EDIT_OPERATIONS:
        raise ValueError(f"Too many edit operations: {len(operations)} > {MAX_EDIT_OPERATIONS}")
    clean: list[dict[str, Any]] = []
    seen_searches: set[str] = set()
    for index, operation in enumerate(operations, 1):
        if not isinstance(operation, dict):
            raise ValueError(f"Operation #{index} is not an object")
        search = operation.get("search")
        replace = operation.get("replace")
        expected = operation.get("expected_occurrences", 1)
        if not isinstance(search, str) or not search:
            raise ValueError(f"Operation #{index}: search must be a non-empty exact substring")
        if not isinstance(replace, str):
            raise ValueError(f"Operation #{index}: replace must be a string")
        if not isinstance(expected, int) or expected < 1:
            raise ValueError(f"Operation #{index}: expected_occurrences must be >= 1")
        if len(search) > MAX_EDIT_SEARCH_CHARS:
            raise ValueError(f"Operation #{index}: search block is too large")
        if search in seen_searches:
            raise ValueError(f"Operation #{index}: duplicate search block")
        if search == replace:
            raise ValueError(f"Operation #{index}: replacement produces no change")
        seen_searches.add(search)
        clean.append({"search": search, "replace": replace, "expected_occurrences": expected})
    return {"action": "modify", "file": file_path, "content": None, "operations": clean}

def apply_edit_plan(project_root: Path, run_dir: Path, task: dict[str, Any], edit_plan: dict[str, Any]) -> dict[str, Any]:
    target = resolve_repo_path(project_root, task["file"])
    task_attempt = int(task.get("attempt", 1))
    if edit_plan["action"] == "create":
        if target.exists():
            raise FileExistsError(f"Create task target already exists: {task['file']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, edit_plan["content"])
        return {
            "action": "create",
            "file": task["file"],
            "backup": None,
            "operations": [],
            "before_sha256": None,
            "after_sha256": sha256_bytes(target.read_bytes()),
        }

    if not target.exists():
        raise FileNotFoundError(f"Target file does not exist: {task['file']}")
    before_bytes = target.read_bytes()
    before = before_bytes.decode("utf-8-sig")
    updated = before
    operation_results: list[dict[str, Any]] = []

    # Validate every operation against the same pre-edit file first. This prevents
    # one replacement from silently changing the search surface of a later operation.
    for index, operation in enumerate(edit_plan["operations"], 1):
        search = operation["search"]
        replace = operation["replace"]
        expected = operation["expected_occurrences"]
        if search == before:
            raise ValueError("Search block equals the entire file; whole-file rewrite is forbidden")
        if len(search) > max(MAX_EDIT_SEARCH_CHARS, int(len(before) * 0.50)):
            raise ValueError(f"Operation #{index} is too large for a surgical edit")
        count = before.count(search)
        if count != expected:
            raise ValueError(f"Operation #{index}: expected {expected} occurrence(s), found {count} in {task['file']}")
        operation_results.append({
            "index": index,
            "expected_occurrences": expected,
            "matched_occurrences": count,
            "search_sha256": sha256_bytes(search.encode("utf-8")),
            "replace_sha256": sha256_bytes(replace.encode("utf-8")),
        })

    # Reject overlapping search blocks because edit ordering would become ambiguous.
    searches = [operation["search"] for operation in edit_plan["operations"]]
    for i, first in enumerate(searches):
        for j, second in enumerate(searches):
            if i < j and (first in second or second in first):
                raise ValueError(f"Operations #{i + 1} and #{j + 1} have overlapping search blocks")

    for operation in edit_plan["operations"]:
        updated = updated.replace(operation["search"], operation["replace"], operation["expected_occurrences"])

    if updated == before:
        raise ValueError("Editor produced no change")

    before_sha = sha256_bytes(before_bytes)
    backup = backup_file(project_root, run_dir, target, task_id=task["id"], attempt=task_attempt)
    atomic_write_text(target, updated)
    after_sha = sha256_bytes(target.read_bytes())
    return {
        "action": "modify",
        "file": task["file"],
        "backup": backup,
        "operations": operation_results,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
    }

def diff_from_backup(run_dir: Path, target_file: str, current_path: Path, backup_rel: str | None, max_chars: int = 20000) -> str:
    if not backup_rel:
        return f"Created new file: {target_file}\n"
    backup_path = run_dir / backup_rel
    if not backup_path.exists() or not current_path.exists():
        return "Diff unavailable."
    before = backup_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    after = current_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(before, after, fromfile=f"a/{target_file}", tofile=f"b/{target_file}"))
    return diff[:max_chars]


def refresh_manifest_target(project_root: Path, manifest: dict[str, Any], target_file: str) -> dict[str, Any]:
    target = resolve_repo_path(project_root, target_file)
    if not target.exists() or not target.is_file():
        manifest["files"] = [item for item in manifest.get("files", []) if item.get("path") != target_file]
        return manifest
    text_content = read_text_file(target)
    language = infer_language(target, text_content)
    symbols, references, imports, calls = extract_symbols_and_edges(text_content, language)
    raw = target.read_bytes()
    updated_item = {
        "path": target_file,
        "language": language,
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "symbols": symbols,
        "references": references,
        "imports": imports,
        "calls": calls,
        "parse_engine": "tree-sitter" if tree_sitter_parser(language) is not None else "fallback",
    }
    files = list(manifest.get("files", []))
    replaced = False
    for index, item in enumerate(files):
        if item.get("path") == target_file:
            files[index] = updated_item
            replaced = True
            break
    if not replaced:
        files.append(updated_item)
    manifest["files"] = sorted(files, key=lambda item: item.get("path", ""))
    # Keep symbol map/file graph internally consistent enough for subsequent edits.
    file_set = {item["path"] for item in manifest["files"]}
    symbol_index: dict[str, list[str]] = {}
    for item in manifest["files"]:
        for symbol in item.get("symbols", []):
            symbol_index.setdefault(symbol.get("name", ""), []).append(item["path"])
    graph = {item["path"]: dict(manifest.get("file_graph", {}).get(item["path"], {})) for item in manifest["files"]}
    graph[target_file] = {}
    for name in set(references):
        for candidate in symbol_index.get(name, []):
            if candidate != target_file:
                graph[target_file][candidate] = graph[target_file].get(candidate, 0.0) + 1.0
    for import_item in imports:
        for candidate in resolve_import_targets(project_root, target_file, import_item.get("text", ""), file_set):
            if candidate != target_file:
                graph[target_file][candidate] = graph[target_file].get(candidate, 0.0) + 2.0
    for source in graph:
        for missing in list(graph[source]):
            if missing not in file_set:
                del graph[source][missing]
    manifest["file_graph"] = graph
    manifest["page_rank"] = compute_pagerank(graph)
    manifest["repo_map"] = build_repo_map(manifest["files"], manifest["page_rank"], max_symbols=MAX_REPO_MAP_SYMBOLS)
    manifest["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return manifest

# ==========================================
# 11. EDITOR NODE
# ==========================================
def editor_node(state: AgentState) -> AgentState:
    project_root = Path(state["project_address"]).resolve()
    run_dir = Path(state["run_dir"])
    manifest = deepcopy(state["manifest"])
    stack = list(state.get("plan_stack", []))
    current_task = deepcopy(state.get("current_task")) if state.get("current_task") else None

    if current_task and current_task.get("review_status") == "PASS":
        current_task = None
    if current_task is None:
        if not stack:
            return {"editor_status": "EVALUATE", "workflow_status": "COMPLETED", "current_task": None, "issue_report": ""}
        current_task = deepcopy(stack.pop())  # strict LIFO: pop exactly one task

    task_id = str(current_task["id"])
    target_file = str(current_task["file"])
    target_path = resolve_repo_path(project_root, target_file)
    target_manifest = manifest_target(manifest, target_file)
    target_language = target_manifest.get("language", infer_language(target_path))
    if current_task.get("action") == "modify" and not target_path.exists():
        raise FileNotFoundError(f"Planner target vanished: {target_file}")
    if current_task.get("action") == "create" and target_path.exists():
        # A previous attempt may already have created it; make the state explicit rather than overwriting it.
        raise FileExistsError(f"Create target already exists: {target_file}")

    attempt_map = dict(state.get("task_attempts", {}))
    attempt = attempt_map.get(task_id, 0) + 1
    attempt_map[task_id] = attempt
    current_task["attempt"] = attempt

    current_code = read_text_file(target_path) if target_path.exists() else ""
    if target_path.exists() and len(current_code.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError(f"Target file exceeds MAX_FILE_BYTES: {target_file}")

    retrieval = build_layered_context_with_rag(
        project_root,
        run_dir,
        manifest,
        f"{state['user_request']}\n{state['requirements']}\nTASK {json.dumps(current_task, ensure_ascii=False)}",
        target_file=target_file,
        target_symbol=current_task.get("function", ""),
        include_full_target=target_path.exists() and target_path.stat().st_size <= MAX_LLM_FILE_BYTES,
    )
    layered_context = render_retrieval_context(retrieval, include_target=False)

    lsp_context = ""
    baseline_lsp: dict[str, Any] = {}
    if ENABLE_LSP and target_path.exists() and target_language in LANGUAGE_LSP_IDS:
        range_info = symbol_range(manifest, target_file, current_task.get("function", ""))
        line = max(0, (range_info[0] - 1) if range_info else 0)
        lsp_result = lsp_analyze_file(project_root, target_file, target_language, current_code, line=line, character=0)
        baseline_lsp = {
            "available": bool(lsp_result.get("available")),
            "error_messages": [item for item in lsp_result.get("diagnostics", []) if int(item.get("severity", 1)) == 1],
        }
        lsp_context = render_lsp_context(lsp_result)

    language_rules = LANGUAGE_SYSTEM_PROMPTS.get(
        target_language,
        "Use the repository's existing syntax, toolchain, dependencies, formatting, error handling and API conventions.",
    )
    system_prompt = f"""
You are the Editor Agent in a production local repository-editing system.

TARGET LANGUAGE:
{target_language}

LANGUAGE-SPECIFIC RULES:
{language_rules}

GLOBAL SURGICAL-EDIT RULES:
- Modify ONLY the requested repository file.
- Existing files MUST use exact literal search/replace operations.
- Every search string MUST be copied verbatim from the CURRENT SOURCE shown to you.
- All searches are validated against the SAME pre-edit file before any operation is applied.
- expected_occurrences MUST equal the exact literal count in the current file.
- Do NOT use regex, line-number edits, placeholders, broad refactors, or full-file rewrites.
- Keep each search block as small as safely possible while uniquely identifying the intended code.
- A search block may not equal the entire file.
- New files may be generated completely only when the task action is explicitly create.
- Do not modify imports/configuration/other functions unless the task explicitly requires that change.
- Preserve unrelated whitespace and code exactly whenever possible.
- Return ONLY valid JSON. No Markdown fences and no explanation.

LSP CONTEXT:
{lsp_context or 'Unavailable.'}
"""

    shown_source = current_code
    if current_code and len(current_code.encode("utf-8")) > MAX_LLM_FILE_BYTES:
        shown_source = extract_target_context(current_code, target_manifest, current_task.get("function", ""))
    user_prompt = f"""
TASK:
{json.dumps(current_task, indent=2, ensure_ascii=False)}

USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements']}

PREVIOUS ISSUE REPORT:
{state.get('issue_report') or 'None'}

COMPACT MANIFEST:
{compact_manifest_for_llm(manifest)}

LAYERED CONTEXT:
{layered_context}

TARGET FILE: {target_file}
CURRENT SOURCE:
{shown_source}

OUTPUT CONTRACT:
For an existing file return:
{{
  "action": "modify",
  "file": "{target_file}",
  "operations": [
    {{
      "search": "EXACT CURRENT SUBSTRING",
      "replace": "TARGETED REPLACEMENT",
      "expected_occurrences": 1
    }}
  ]
}}

For a new file return:
{{
  "action": "create",
  "file": "{target_file}",
  "content": "COMPLETE NEW FILE CONTENT"
}}

For this attempt, fix the requested task and any reviewer issue, but nothing else.
"""
    write_log(run_dir, f"Editor Node [{task_id}] - SYSTEM PROMPT", system_prompt)
    write_log(run_dir, f"Editor Node [{task_id}] - USER PROMPT", user_prompt)

    started = time.perf_counter()
    raw = ""
    edit_plan = None
    generation_error: Exception | None = None
    for generation_attempt in range(1, MAX_EDITOR_GENERATION_ATTEMPTS + 1):
        try:
            generation_prompt = user_prompt
            if generation_attempt > 1:
                generation_prompt += (
                    f"\n\nREPAIR ATTEMPT {generation_attempt}: Your previous response was not an applicable edit. "
                    "Return ONLY the required JSON object using exact text copied from the CURRENT SOURCE. "
                    f"Previous response:\n{raw[:8000]}"
                )
            raw = invoke_llm(generation_prompt, system_prompt=system_prompt)
            edit_plan = validate_edit_plan(safe_json_load(raw), current_task)
            generation_error = None
            break
        except Exception as exc:
            generation_error = exc
            write_log(
                run_dir,
                f"Editor Node [{task_id}] - GENERATION ATTEMPT {generation_attempt}",
                f"{type(exc).__name__}: {exc}\nRAW:\n{raw[:12000]}",
            )

    try:
        if generation_error is not None or edit_plan is None:
            raise generation_error or ValueError("Editor did not produce an applicable edit plan")
        result = apply_edit_plan(project_root, run_dir, current_task, edit_plan)
    except Exception as exc:
        duration = time.perf_counter() - started
        task_attempts = dict(state.get("task_attempts", {}))
        attempt_count = int(task_attempts.get(task_id, attempt))
        issue = (
            f"# Editor Issue — {task_id}\n\n"
            f"## Target\n`{target_file}` — `{current_task.get('function', '')}`\n\n"
            f"## Attempt\n{attempt}\n\n"
            f"## Problem\n`{type(exc).__name__}: {exc}`\n\n"
            "## Required Action\nRegenerate a minimal exact search/replace JSON edit from the current source.\n"
        )
        atomic_write_text(run_dir / "issue.md", issue)
        durations = dict(state.get("task_durations", {}))
        durations[task_id] = round(durations.get(task_id, 0.0) + duration, 4)
        return {
            "plan_stack": stack,
            "current_task": current_task,
            "issue_report": issue,
            "review_attempts": attempt_count,
            "iterations": state.get("iterations", 0) + 1,
            "task_attempts": task_attempts,
            "task_durations": durations,
            "editor_status": "RETRY" if attempt_count < MAX_REVIEW_ATTEMPTS else "FAILED",
            "workflow_status": "RUNNING" if attempt_count < MAX_REVIEW_ATTEMPTS else "FAILED",
            "failure_reason": issue if attempt_count >= MAX_REVIEW_ATTEMPTS else state.get("failure_reason", ""),
        }
    duration = time.perf_counter() - started
    current_task["review_status"] = "PENDING"
    current_task["last_edit_seconds"] = round(duration, 4)
    current_task["last_edit"] = result
    file_attempts = dict(state.get("file_attempts", {}))
    file_attempts[target_file] = file_attempts.get(target_file, 0) + 1
    history = list(state.get("task_history", []))
    history.append({
        "task_id": task_id,
        "attempt": attempt,
        "file": target_file,
        "function": current_task.get("function", ""),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(duration, 4),
        "edit": result,
        "status": "EDITED_PENDING_REVIEW",
    })
    durations = dict(state.get("task_durations", {}))
    durations[task_id] = round(durations.get(task_id, 0.0) + duration, 4)
    write_log(run_dir, f"Editor Node [{task_id}] - APPLIED EDIT", json.dumps(result, indent=2, ensure_ascii=False))
    return {
        "plan_stack": stack,
        "current_task": current_task,
        "issue_report": "",
        "review_attempts": 0,
        "iterations": state.get("iterations", 0) + 1,
        "file_attempts": file_attempts,
        "task_attempts": attempt_map,
        "task_durations": durations,
        "task_history": history,
        "last_edit": result,
        "editor_status": "REVIEW",
        "workflow_status": "RUNNING",
        "retrieval_context": layered_context,
        "lsp_context": lsp_context,
        "baseline_lsp": baseline_lsp,
    }

# ==========================================
# 12. REVIEWER NODE
# ==========================================
def local_validation(project_root: Path, target_file: str, manifest: dict[str, Any]) -> tuple[bool, str]:
    target = resolve_repo_path(project_root, target_file)
    if not target.exists():
        return False, f"Target file does not exist: {target_file}"
    try:
        text_content = read_text_file(target)
    except Exception as exc:
        return False, str(exc)

    language = manifest_target(manifest, target_file).get("language", infer_language(target, text_content))
    checks: list[str] = []

    if language == "Python":
        try:
            compile(text_content, str(target), "exec")
            checks.append("Python compile()")
        except SyntaxError as exc:
            return False, f"Python syntax error: {exc}"

    if target.suffix.lower() == ".json":
        try:
            json.loads(text_content)
            checks.append("JSON parse")
        except json.JSONDecodeError as exc:
            return False, f"JSON syntax error: {exc}"

    if target.suffix.lower() in {".js", ".mjs", ".cjs"}:
        node = shutil.which("node")
        if node:
            ok, output = run_optional_command([node, "--check", str(target)], project_root, 30)
            if not ok:
                return False, f"Node syntax check failed: {output[:12000]}"
            checks.append("Node --check")

    if target.suffix.lower() in {".ts", ".tsx"}:
        tsc = shutil.which("tsc")
        config = project_root / "tsconfig.json"
        if tsc and config.exists():
            ok, output = run_optional_command([tsc, "--noEmit", "--pretty", "false", "--project", str(config)], project_root, 90)
            if not ok:
                return False, f"TypeScript project check failed: {output[:12000]}"
            checks.append("tsc --noEmit")

    c_like = language in {"C", "C/C++ Header", "C++", "C++ Header", "Objective-C", "Objective-C++"}
    if c_like:
        compiler = shutil.which("cc") if language in {"C", "C/C++ Header"} else shutil.which("c++")
        if compiler:
            cmd = [compiler, "-fsyntax-only", "-I", str(project_root)]
            if language in {"C/C++ Header", "C++ Header", "C++", "Objective-C++"} or target.suffix.lower() in {".hh", ".hpp", ".hxx"}:
                cmd += ["-x", "c++"]
            elif target.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}:
                cmd += ["-x", "c"]
            cmd.append(str(target))
            ok, output = run_optional_command(cmd, project_root, 60)
            if not ok:
                return False, f"C/C++ syntax check failed: {output[:12000]}"
            checks.append(f"{Path(compiler).name} -fsyntax-only")

    if RUN_PROJECT_TESTS:
        if language == "Go" and (project_root / "go.mod").exists() and shutil.which("go"):
            ok, output = run_optional_command(["go", "test", "./..."], project_root, 120)
            if not ok:
                return False, f"go test failed: {output[:12000]}"
            checks.append("go test ./...")
        elif language == "Rust" and (project_root / "Cargo.toml").exists() and shutil.which("cargo"):
            ok, output = run_optional_command(["cargo", "check"], project_root, 120)
            if not ok:
                return False, f"cargo check failed: {output[:12000]}"
            checks.append("cargo check")

    if ENABLE_LSP and language in LANGUAGE_LSP_IDS:
        result = lsp_analyze_file(project_root, target_file, language, text_content)
        if result.get("available"):
            diagnostics = result.get("diagnostics", [])
            errors = [item for item in diagnostics if int(item.get("severity", 1)) == 1]
            checks.append(f"LSP diagnostics ({len(errors)} error(s) present)" if errors else "LSP diagnostics")

    return True, "Local validation passed: " + (", ".join(checks) if checks else "no installed deterministic validator")

def reviewer_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    project_root = Path(state["project_address"])
    manifest = deepcopy(state["manifest"])
    task = deepcopy(state.get("current_task")) if state.get("current_task") else None
    if not task:
        return {"editor_status": "EVALUATE"}

    target_file = task["file"]
    target_path = resolve_repo_path(project_root, target_file)
    try:
        target_code = read_text_file(target_path)
    except Exception as exc:
        issue = f"# Review Issue — {task['id']}\n\nUnable to read `{target_file}` after edit: {exc}\n"
        atomic_write_text(run_dir / "issue.md", issue)
        return {"issue_report": issue, "review_attempts": state.get("review_attempts", 0) + 1, "workflow_status": "RUNNING"}

    actual_hash = sha256_bytes(target_path.read_bytes())
    expected_hash = state.get("last_edit", {}).get("after_sha256")
    if expected_hash and actual_hash != expected_hash:
        issue = (
            f"# Review Issue — {task['id']}\n\n"
            f"Target `{target_file}` changed unexpectedly after the Editor wrote it.\n"
            f"Expected hash: `{expected_hash}`\nObserved hash: `{actual_hash}`\n"
        )
        atomic_write_text(run_dir / "issue.md", issue)
        return {"issue_report": issue, "review_attempts": state.get("review_attempts", 0) + 1, "workflow_status": "RUNNING"}

    # Refresh only the touched file plus graph metadata so subsequent tasks see new symbols/imports.
    try:
        manifest = refresh_manifest_target(project_root, manifest, target_file)
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        atomic_write_text(run_dir / "manifest.json", manifest_json)
    except Exception as exc:
        write_log(run_dir, f"Reviewer Manifest Refresh Warning [{task['id']}]", str(exc))
        manifest_json = state["manifest_json"]

    local_ok, local_feedback = local_validation(project_root, target_file, manifest)
    if ENABLE_LSP:
        try:
            language = manifest_target(manifest, target_file).get("language", infer_language(target_path, target_code))
            if language in LANGUAGE_LSP_IDS:
                current_lsp = lsp_analyze_file(project_root, target_file, language, target_code)
                if current_lsp.get("available"):
                    baseline_errors = state.get("baseline_lsp", {}).get("error_messages", [])
                    current_errors = [item for item in current_lsp.get("diagnostics", []) if int(item.get("severity", 1)) == 1]
                    baseline_keys = {(item.get("range", {}).get("start", {}).get("line"), item.get("message", "")) for item in baseline_errors}
                    new_errors = [item for item in current_errors if (item.get("range", {}).get("start", {}).get("line"), item.get("message", "")) not in baseline_keys]
                    if new_errors:
                        local_ok = False
                        local_feedback += "\nNew LSP errors introduced by this edit: " + json.dumps(new_errors[:12], ensure_ascii=False)
        except Exception as exc:
            write_log(run_dir, f"Reviewer LSP Comparison Warning [{task['id']} ]", str(exc))
    diff = diff_from_backup(run_dir, target_file, target_path, state.get("last_edit", {}).get("backup"))
    retrieval = build_layered_context_with_rag(
        project_root,
        run_dir,
        manifest,
        f"{state['user_request']}\n{state['requirements']}\n{json.dumps(task, ensure_ascii=False)}",
        target_file=target_file,
        target_symbol=task.get("function", ""),
        include_full_target=target_path.exists() and target_path.stat().st_size <= MAX_LLM_FILE_BYTES,
    )
    context = render_retrieval_context(retrieval, include_target=False)
    prompt = f"""
You are the Reviewer Agent for one surgical repository edit.

OVERARCHING USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements']}

COMPACT MANIFEST:
{compact_manifest_for_llm(manifest)}

TASK BEING REVIEWED:
{json.dumps(task, indent=2, ensure_ascii=False)}

TARGET FILE: {target_file}

LOCAL VALIDATION:
{local_feedback}

ACTUAL EDIT DIFF:
{diff}

CURRENT TARGET SOURCE:
{task_source_context(target_code, task, manifest_target(manifest, target_file))}

LAYERED CONTEXT:
{context}

Review ONLY this task. Confirm the requested behavior is implemented, the target symbol/section is appropriate, the change is consistent with language/framework/architecture, and no unrelated behavior was changed.

If fully correct, return ONLY:
PASS

Otherwise return ONLY JSON:
{{
  "status": "FAIL",
  "summary": "concise issue",
  "replace": "exact problematic source snippet when available",
  "with": "exact corrected source snippet when available",
  "reason": "specific task/manifest/validation violation"
}}
"""
    write_log(run_dir, f"Reviewer Node [{task['id']}] - PROMPT", prompt)
    review_output = invoke_llm(prompt) if local_ok else json.dumps({
        "status": "FAIL", "summary": local_feedback, "replace": "", "with": "", "reason": local_feedback
    }, ensure_ascii=False)
    write_log(run_dir, f"Reviewer Node [{task['id']}] - OUTPUT", review_output)

    if reviewer_says_pass(review_output):
        approved = deepcopy(task)
        approved["review_status"] = "PASS"
        approved["approved_at"] = datetime.now().isoformat(timespec="seconds")
        history = list(state.get("task_history", []))
        for item in reversed(history):
            if item.get("task_id") == task["id"] and item.get("status") == "EDITED_PENDING_REVIEW":
                item["status"] = "PASS"
                break
        completed = list(state.get("completed_tasks", []))
        if task["id"] not in completed:
            completed.append(task["id"])
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        return {
            "manifest": manifest,
            "manifest_json": manifest_json,
            "repo_map": json.dumps(manifest.get("repo_map", []), ensure_ascii=False),
            "current_task": approved,
            "issue_report": "",
            "review_attempts": 0,
            "completed_tasks": completed,
            "task_history": history,
            "workflow_status": "RUNNING",
            "manifest_refresh_count": state.get("manifest_refresh_count", 0) + 1,
        }

    try:
        structured = safe_json_load(review_output)
        if not isinstance(structured, dict):
            raise ValueError("reviewer JSON is not an object")
        issue = (
            f"# Review Issue — {task['id']}\n\n"
            f"## Target\n`{target_file}` — `{task.get('function', '')}`\n\n"
            f"## Summary\n{structured.get('summary', 'Reviewer found an issue.')}\n\n"
            f"## Surgical Change\n**Replace:**\n```text\n{structured.get('replace', '')}\n```\n\n"
            f"**With:**\n```text\n{structured.get('with', '')}\n```\n\n"
            f"## Reason\n{structured.get('reason', '')}\n"
        )
    except Exception:
        issue = f"# Review Issue — {task['id']}\n\n{review_output}\n"
    atomic_write_text(run_dir / "issue.md", issue)
    return {
        "manifest": manifest,
        "manifest_json": json.dumps(manifest, indent=2, ensure_ascii=False),
        "repo_map": json.dumps(manifest.get("repo_map", []), ensure_ascii=False),
        "current_task": task,
        "issue_report": issue,
        "review_attempts": state.get("review_attempts", 0) + 1,
        "workflow_status": "RUNNING",
    }

# ==========================================
# 13. ROUTING
# ==========================================
def route_editor(state: AgentState) -> str:
    status = state.get("editor_status")
    if status == "EVALUATE" and not state.get("plan_stack") and state.get("current_task") is None:
        return "evaluate"
    if status == "FAILED":
        return "failure"
    if status == "RETRY":
        return "retry"
    return "review"

def route_reviewer(state: AgentState) -> str:
    if not state.get("issue_report"):
        return "next"
    if state.get("review_attempts", 0) >= MAX_REVIEW_ATTEMPTS:
        return "failure"
    return "fix"

def failure_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    reason = state.get("failure_reason") or state.get("issue_report") or "Workflow stopped after maximum retry attempts."
    task = state.get("current_task") or {}
    backup_rel = state.get("last_edit", {}).get("backup")
    target_file = task.get("file")
    if backup_rel and target_file:
        target = resolve_repo_path(Path(state["project_address"]), target_file)
        backup = run_dir / backup_rel
        if backup.exists():
            try:
                atomic_write_text(target, backup.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                reason += f"\nRollback failed: {exc}"
    atomic_write_text(run_dir / "workflow_failure.md", f"# Workflow Failure\n\n{reason}\n")
    write_log(run_dir, "Workflow Failure", reason)
    return {
        "workflow_status": "FAILED",
        "failure_reason": reason,
        "editor_status": "EVALUATE",
        "plan_stack": [],
        "current_task": None,
    }

# ==========================================
# 14. EVALUATION (DeepEval + timing/task matrix)
# ==========================================
if DEEPEVAL_AVAILABLE:

    class OllamaEvalModel(DeepEvalBaseLLM):
        def __init__(self, chat_model: ChatOllama, name: str):
            self._chat = chat_model
            self._name = name

        def load_model(self):
            return self._chat

        def generate(self, prompt: str) -> str:
            return response_text(self._chat.invoke(prompt))

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
                "Using the requirements, manifest, task history, and actual output as ground truth: does the final changed "
                "code plausibly implement the requested behavior without obvious runtime, API, reference, or logic bugs?"
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        ),
        GEval(
            name="Completeness",
            criteria=(
                "Using the requirements and manifest as the specification: does the final changed code implement the requested "
                "functionality with real logic and without TODO-only or empty stubs?"
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
        f"**Workflow status:** {overall.get('workflow_status', 'UNKNOWN')}",
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
            f"- **{task['task_id']}** — {task['file']} — `{task['duration_seconds']:.3f}s` — "
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
            f"<tr><td>{i + 1}</td><td>{html.escape(str(record.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(cfg.get('model', '')))}</td><td>{html.escape(str(cfg.get('num_ctx', '')))}</td>"
            f"<td>{html.escape(str(record.get('project_name', '')))}</td><td>{record.get('total_tasks', '')}</td>"
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
    tasks_completed = len(set(state.get("completed_tasks", [])))

    context = (
        f"USER REQUEST:\n{state['user_request']}\n\nREQUIREMENTS:\n{state['requirements']}\n\n"
        f"MANIFEST:\n{compact_manifest_for_llm(state['manifest'], MAX_MANIFEST_PROMPT_CHARS)}\n\nTASK HISTORY:\n"
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
    counted_tasks = tasks_completed if tasks_completed else total_tasks
    time_per_task = elapsed_seconds / counted_tasks if counted_tasks else 0.0
    overall = {
        "files_passed": f"{len(changed_files)}/{len(changed_files)}",
        "changed_files": changed_files,
        "total_tasks": total_tasks,
        "tasks_completed": tasks_completed,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "time_per_task_seconds": round(time_per_task, 4),
        "task_timing": task_timing,
        "workflow_status": state.get("workflow_status", "UNKNOWN"),
        "failure_reason": state.get("failure_reason", ""),
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
    atomic_write_text(run_dir / "evaluation.json", json.dumps(evaluation, indent=2, ensure_ascii=False))
    atomic_write_text(run_dir / "evaluation.md", render_evaluation_markdown(overall, report))
    write_log(run_dir, "Evaluator", json.dumps(overall, indent=2, ensure_ascii=False))

    benchmark_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "project_name": overall["project_name"],
        "project_address": str(project_root),
        "user_request": state.get("user_request", "")[:300],
        "llm_config": {
            "model": OLLAMA_MODEL,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
        },
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
    atomic_write_text(benchmark_history_path().parent / "benchmark_history.html", render_benchmark_html(history))

    print(f"   Custom score avg: {overall['custom_score_avg']}/1.0")
    if "deepeval_avg_score" in overall:
        print(f"   DeepEval score avg: {overall['deepeval_avg_score']}/1.0")
    print(f"   ⏱️ Total time: {overall['elapsed_seconds']:.3f}s")
    print(f"   🧩 Total tasks: {overall['total_tasks']}")
    print(f"   ✅ Tasks completed: {overall['tasks_completed']}")
    print(f"   ⏱️ Time/task: {overall['time_per_task_seconds']:.3f}s")
    print(f"   📈 Benchmark: {len(history)} run(s) tracked — see benchmark_history.jsonl / benchmark_history.html")
    return {"elapsed_seconds": elapsed_seconds, "evaluation": evaluation}


# ==========================================
# 15. GRAPH
# ==========================================
def build_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("requirements", requirement_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("failure", failure_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("requirements")
    workflow.add_edge("requirements", "analyzer")
    workflow.add_edge("analyzer", "planner")
    workflow.add_edge("planner", "editor")

    workflow.add_conditional_edges(
        "editor",
        route_editor,
        {
            "review": "reviewer",
            "retry": "editor",
            "evaluate": "evaluator",
            "failure": "failure",
        },
    )
    workflow.add_conditional_edges(
        "reviewer",
        route_reviewer,
        {
            "fix": "editor",
            "next": "editor",
            "failure": "failure",
        },
    )
    workflow.add_edge("failure", "evaluator")
    workflow.add_edge("evaluator", END)
    return workflow.compile()


app = build_workflow()


# ==========================================
# 16. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 100)
    print("Dynamic Local Repository Editor — LangGraph + Ollama")
    print("AST + PageRank RepoMap + Local RAG + LSP + Layered Context + Surgical Edits")
    print("=" * 100)
    print(f"Ollama model: {OLLAMA_MODEL}")
    print(f"Ollama context: {OLLAMA_NUM_CTX} | max output: {OLLAMA_NUM_PREDICT} | temperature: {OLLAMA_TEMPERATURE}")

    project_address = input("\nPath to existing project/repository:\n> ").strip()
    user_request = input("\nWhat change should be made?\n> ").strip()
    project_path = Path(project_address).expanduser().resolve()
    if not project_path.is_dir():
        raise SystemExit(f"Project directory does not exist: {project_path}")
    if not user_request:
        raise SystemExit("Change request must not be empty")

    requested_run_base = Path(os.getenv("SDLC_RUN_DIR", str(Path.cwd() / "SDLC_Runs"))).expanduser().resolve()
    try:
        requested_run_base.relative_to(project_path)
        requested_run_base = project_path.parent / ".SDLC_Runs"
    except ValueError:
        pass
    run_dir = requested_run_base / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    write_log(run_dir, "SYSTEM", json.dumps({
        "project_address": str(project_path),
        "user_request": user_request,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_num_ctx": OLLAMA_NUM_CTX,
        "ollama_num_predict": OLLAMA_NUM_PREDICT,
        "ollama_temperature": OLLAMA_TEMPERATURE,
        "embedding_model": OLLAMA_EMBED_MODEL,
        "rag_enabled": ENABLE_RAG,
        "lsp_enabled": ENABLE_LSP,
    }, indent=2, ensure_ascii=False))

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
        "repo_map": "",
        "retrieval_context": "",
        "lsp_context": "",
        "baseline_lsp": {},
        "plan_stack": [],
        "current_task": None,
        "issue_report": "",
        "review_attempts": 0,
        "iterations": 0,
        "file_attempts": {},
        "task_attempts": {},
        "task_durations": {},
        "task_history": [],
        "completed_tasks": [],
        "total_tasks": 0,
        "elapsed_seconds": 0.0,
        "editor_status": "READY",
        "workflow_status": "RUNNING",
        "last_edit": {},
        "evaluation": {},
    }

    try:
        # A task can consume several graph transitions (Editor -> Reviewer, plus reviewer/editor retries).
        # Size the recursion budget from the maximum planner task count rather than using a small fixed default.
        recursion_limit = max(MAX_GRAPH_STEPS, (MAX_PLANNER_TASKS * (MAX_REVIEW_ATTEMPTS + 5)) + 20)
        final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})
        if isinstance(final_state, dict):
            print(f"\n🔚 Final workflow status: {final_state.get('workflow_status', 'UNKNOWN')}")
            print(f"🧩 Completed tasks: {len(final_state.get('completed_tasks', []))}/{final_state.get('total_tasks', 0)}")
            print(f"📝 Last task: {(final_state.get('current_task') or {}).get('id', 'none')}")
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        failure = (
            f"# Workflow Failure\n\n"
            f"**Exception:** `{type(exc).__name__}`\n\n"
            f"**Message:** {exc}\n\n"
            f"**Elapsed seconds:** {elapsed:.4f}\n"
        )
        atomic_write_text(run_dir / "workflow_failure.md", failure)
        write_log(run_dir, "SYSTEM FAILURE", failure)
        raise

    print("\n🎉 Workflow Complete.")
    print(f"📊 Evaluation: {run_dir / 'evaluation.md'}")
    print(f"🧾 Manifest: {run_dir / 'manifest.json'}")
    print(f"📚 Plan stack: {run_dir / 'plan_stack.json'}")
    print(f"🧠 RAG index: {run_dir / 'rag_index.json'}")
    print(f"💾 Backups: {run_dir / 'backups'}")
    print(f"📈 Benchmark history: {benchmark_history_path().parent / 'benchmark_history.html'}")
