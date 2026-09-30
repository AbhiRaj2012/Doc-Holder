from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
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
MAX_LLM_FILE_BYTES = int(os.getenv("MAX_LLM_FILE_BYTES", str(350 * 1024)))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "85000"))
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
    text = data.decode("utf-8")
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
        temp_path.write_text(content, encoding="utf-8")
        if path.exists():
            try:
                shutil.copymode(path, temp_path)
            except OSError:
                pass
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


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
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output


# ==========================================
# 3. LANGUAGE / AST / REPOSITORY ANALYSIS
# ==========================================
def infer_language(path: Path, text: str = "") -> str:
    language = LANGUAGE_BY_EXT.get(path.suffix, "Unknown")
    if path.suffix.lower() in {".h", ".hh", ".hpp", ".hxx"}:
        sample = text[:12000]
        if any(token in sample for token in ("std::", "template<", "class ", "namespace ", "#include <vector>", "#include <string>")):
            return "C++ Header"
        return "C/C++ Header"
    if path.name == "Dockerfile":
        return "Dockerfile"
    if path.name in {"Makefile", "CMakeLists.txt"}:
        return "Build/Config"
    if not language and path.name.endswith(".config"):
        return "Config"
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
    lowered = [path.lower() for path in paths]
    top_level_dirs = sorted({Path(path).parts[0] for path in paths if len(Path(path).parts) > 1})
    signals: set[str] = set()
    patterns = {
        "components": ("/components/",), "controllers": ("/controllers/",), "services": ("/services/",),
        "repositories": ("/repositories/",), "models": ("/models/",), "views": ("/views/",),
        "routes": ("/routes/",), "routers": ("/router/", "routers/"), "handlers": ("/handlers/",),
        "middleware": ("/middleware/",), "utils": ("/utils/", "/util/"), "tests": ("/tests/", "/test/"),
        "src-layout": ("/src/",), "include-layout": ("/include/",), "lib-layout": ("/lib/",),
    }
    for label, markers in patterns.items():
        if any(any(marker in f for marker in markers) for f in lowered):
            signals.add(label)
    if any(Path(path).name.lower() in {"main.py", "app.py", "main.go", "main.rs", "main.cpp", "main.c"} for path in paths):
        signals.add("entrypoint")
    if any(path.endswith(".csproj") for path in lowered):
        signals.add("dotnet-project")
    if "manage.py" in {Path(path).name.lower() for path in paths}:
        signals.add("django-conventions")
    if any(Path(path).name.lower() in {"package.json", "pnpm-workspace.yaml", "yarn.lock"} for path in paths):
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


def collect_repository_context(project_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    language_counts: dict[str, int] = {}
    all_paths: set[str] = set()
    total_bytes = 0

    for root, dirs, filenames in os.walk(project_root):
        dirs[:] = sorted(directory for directory in dirs if directory not in IGNORED_DIRS and not is_sensitive_path(Path(directory)))
        for filename in sorted(filenames):
            if len(files) >= MAX_FILES_TO_SCAN:
                break
            path = Path(root) / filename
            if is_sensitive_path(path):
                continue
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

            relative = path.resolve().relative_to(project_root.resolve()).as_posix()
            language = infer_language(path, text)
            symbols, references, imports, calls = extract_symbols_and_edges(text, language)
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
        if len(files) >= MAX_FILES_TO_SCAN:
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
        "schema_version": "4.0",
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
        "total_files_scanned": len(files),
        "total_text_bytes_scanned": total_bytes,
    }
    return manifest


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
    scored = []
    for file_item in manifest.get("files", []):
        symbol_names = " ".join(symbol["name"] for symbol in file_item.get("symbols", []))
        haystack = f"{file_item['path']} {file_item['language']} {symbol_names}"
        lexical = lexical_similarity(query, haystack)
        pagerank = float(manifest.get("page_rank", {}).get(file_item["path"], 0.0))
        score = 0.70 * lexical + 0.30 * pagerank
        if score > 0:
            scored.append({
                "file": file_item["path"],
                "language": file_item["language"],
                "score": round(score, 6),
                "pagerank": round(pagerank, 6),
                "symbols": [symbol["name"] for symbol in file_item.get("symbols", [])[:25]],
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
            text = read_text_file(path)
        except Exception:
            continue
        lines = text.splitlines()
        symbols = file_item.get("symbols", [])
        if symbols:
            for symbol in symbols:
                if len(chunks) >= max_chunks:
                    break
                start = max(1, int(symbol.get("start_line", 1)) - 4)
                end = min(len(lines), int(symbol.get("end_line", start)) + 4)
                body = "\n".join(lines[start - 1:end])
                if not body.strip():
                    continue
                chunks.append({
                    "id": hashlib.sha1(f"{file_item['path']}:{start}:{end}".encode()).hexdigest()[:16],
                    "path": file_item["path"],
                    "language": file_item["language"],
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
                chunks.append({
                    "id": hashlib.sha1(f"{file_item['path']}:{start}:{end}".encode()).hexdigest()[:16],
                    "path": file_item["path"],
                    "language": file_item["language"],
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
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("model") == OLLAMA_EMBED_MODEL and isinstance(payload.get("records"), list):
                    index.records = payload["records"]
                    existing = {record.get("id") for record in index.records}
                    pending = [chunk for chunk in chunks if chunk["id"] not in existing]
                else:
                    pending = chunks
            else:
                pending = chunks

            for offset in range(0, len(pending), RAG_BATCH_SIZE):
                batch = pending[offset:offset + RAG_BATCH_SIZE]
                vectors = embedder.embed_documents([item["text"] for item in batch])
                for item, vector in zip(batch, vectors):
                    index.records.append({**item, "vector": vector})
                index._persist()
            if not path.exists():
                index._persist()
            return index
        except Exception as exc:
            raise RuntimeError(f"Local Ollama embedding index unavailable: {exc}") from exc

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
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
        if not self.records:
            return []
        vector = self.embeddings.embed_query(query)
        scored = []
        for record in self.records:
            scored.append({**record, "score": round(self.cosine(vector, record.get("vector", [])), 6)})
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
        root_uri = self.project_root.as_uri()
        self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": self.project_root.name}],
            "capabilities": {
                "workspace": {"workspaceFolders": True},
                "textDocument": {
                    "definition": {"linkSupport": True},
                    "references": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                },
            },
            "clientInfo": {"name": "local-langgraph-editor", "version": "1.0"},
        }, timeout=LSP_TIMEOUT_SECONDS)
        self.notify("initialized", {})

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.notify("shutdown", None)
            self.notify("exit", None)
        except Exception:
            pass
        try:
            self.process.kill()
            self.process.wait(timeout=2)
        except Exception:
            pass
        self.process = None
        self._opened_uri = None

    def _write_message(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("LSP process is not running")
        payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
        self.process.stdin.write(header + payload)
        self.process.stdin.flush()

    def notify(self, method: str, params: Any) -> None:
        self._write_message({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout: float = LSP_TIMEOUT_SECONDS) -> dict[str, Any] | None:
        request_id = self._next_id
        self._next_id += 1
        self._write_message({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        diagnostics: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            message = self._read_message(max(0.05, deadline - time.monotonic()))
            if message is None:
                continue
            if message.get("method") == "textDocument/publishDiagnostics":
                diagnostics.append(message.get("params", {}))
                continue
            if message.get("id") == request_id:
                if diagnostics:
                    message["_diagnostics"] = diagnostics
                return message
        return {"error": {"code": -32002, "message": f"LSP request timed out: {method}"}, "_diagnostics": diagnostics}

    def _read_message(self, timeout: float) -> dict[str, Any] | None:
        if self.process is None or self.process.stdout is None:
            return None
        import selectors

        selector = selectors.DefaultSelector()
        try:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            events = selector.select(timeout)
            if not events:
                return None
            header_lines = []
            content_length = None
            while True:
                line = self.process.stdout.readline()
                if not line:
                    return None
                decoded = line.decode("ascii", errors="replace").strip()
                if not decoded:
                    break
                header_lines.append(decoded)
                if decoded.lower().startswith("content-length:"):
                    content_length = int(decoded.split(":", 1)[1].strip())
            if content_length is None:
                return None
            body = self.process.stdout.read(content_length)
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
        finally:
            selector.close()

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
        payload: dict[str, Any] = {"uri": uri}
        result = self.request("textDocument/documentSymbol", payload)
        definition = self.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": max(0, line), "character": max(0, character)},
        })
        references = self.request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": max(0, line), "character": max(0, character)},
            "context": {"includeDeclaration": True},
        })
        return {
            "document_symbols": result.get("result") if result else None,
            "definition": definition.get("result") if definition else None,
            "references": references.get("result") if references else None,
            "diagnostics": [
                diag
                for response in (result, definition, references)
                if response
                for diag_group in response.get("_diagnostics", [])
                for diag in diag_group.get("diagnostics", [])
            ],
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
        result = client.analyze_document(uri, line=line, character=character)
        client.close_document()
        result["available"] = True
        result["command"] = command
        return result
    except Exception as exc:
        return {"available": False, "reason": f"LSP failure: {exc}", "command": command}
    finally:
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
    # Layer 1: directory overview + languages/frameworks + compact repository map.
    overview = "\n".join([
        f"Project: {manifest.get('project', {}).get('name', project_root.name)}",
        f"Languages: {', '.join(item['name'] for item in manifest.get('languages', [])[:12]) or 'unknown'}",
        f"Frameworks: {', '.join(manifest.get('frameworks', [])) or 'none detected'}",
        f"Architecture: {manifest.get('architecture', {}).get('style', 'unknown')}",
        "Directory overview:",
        manifest.get("directory_tree", "")[:12000],
    ])

    # Layer 2: structural repository map + PageRank/lexical seeds.
    ranked = rank_files_for_query(manifest, query, limit=12)
    structural_map = manifest.get("repo_map", [])
    map_lines = []
    for item in structural_map[:MAX_REPO_MAP_SYMBOLS]:
        map_lines.append(
            f"{item['file']}::{item['name']} [{item['kind']}] lines {item['start_line']}-{item['end_line']} rank={item['pagerank']}"
        )
    structural = "\n".join(map_lines)

    # Layer 3: semantic retrieval only for candidates selected by the task/query.
    rag_results: list[dict[str, Any]] = []
    context = {
        "overview": overview,
        "ranked_files": ranked,
        "repo_map": structural,
        "rag_results": rag_results,
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
            related_paths = []
            for item in ranked:
                path = item["file"]
                if path != target_file and len(related_paths) < 4:
                    related_paths.append(path)
            related_parts = []
            for path in related_paths:
                try:
                    related_manifest = manifest_target(manifest, path)
                    related_source = read_text_file(resolve_repo_path(project_root, path), MAX_LLM_FILE_BYTES)
                    snippets = extract_relevant_symbol_snippets(related_source, related_manifest)
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


def render_retrieval_context(context: dict[str, Any]) -> str:
    ranked = "\n".join(
        f"{item['file']} score={item['score']} symbols={', '.join(item.get('symbols', [])[:12])}"
        for item in context.get("ranked_files", [])
    )
    rag = "\n".join(
        f"{item.get('path')}:{item.get('start_line')}-{item.get('end_line')} score={item.get('score')}\n{item.get('text', '')[:5000]}"
        for item in context.get("rag_results", [])
    )
    return "\n\n".join([
        "[LAYER 1 — OVERVIEW]", context.get("overview", "")[:14000],
        "[LAYER 2 — RANKED FILES]", ranked[:10000],
        "[LAYER 2 — REPO MAP]", context.get("repo_map", "")[:25000],
        "[LAYER 3 — SEMANTIC RETRIEVAL]", rag[:18000],
        "[LAYER 3 — TARGET]", context.get("target", "")[:40000],
        "[LAYER 2/3 — RELATED]", context.get("related", "")[:18000],
    ])[:MAX_CONTEXT_CHARS]


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

    manifest = collect_repository_context(project)
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
    retrieval_context = render_retrieval_context(retrieval)

    prompt = f"""
You are the planning agent for an EXISTING local codebase.

USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements']}

REPOSITORY MANIFEST (ground truth):
{state['manifest_json'][:70000]}

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
        if not target_symbol:
            raise ValueError(f"Task {task_id}: exact function/class/method/section is required")
        if action == "modify" and target_symbol.lower() not in {"module-level imports", "module-level constants", "configuration", "configuration block", "route table", "declaration block"}:
            symbols = {str(symbol.get("name")) for symbol in manifest_target(manifest, file_path).get("symbols", [])}
            if target_symbol not in symbols:
                # Allow a precise section only when it is visibly structural and not an invented function.
                section_keywords = ("imports", "config", "configuration", "declaration", "route table", "registry", "module level", "module-level")
                if not any(keyword in target_symbol.lower() for keyword in section_keywords):
                    raise ValueError(f"Task {task_id}: target symbol '{target_symbol}' is not present in manifest for {file_path}")
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
    action = parsed.get("action")
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
    clean: list[dict[str, Any]] = []
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
        clean.append({"search": search, "replace": replace, "expected_occurrences": expected})
    return {"action": "modify", "file": file_path, "content": None, "operations": clean}


def apply_edit_plan(project_root: Path, run_dir: Path, task: dict[str, Any], edit_plan: dict[str, Any]) -> dict[str, Any]:
    target = resolve_repo_path(project_root, task["file"])
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
    before = read_text_file(target)
    updated = before
    operation_results: list[dict[str, Any]] = []

    for index, operation in enumerate(edit_plan["operations"], 1):
        search = operation["search"]
        replace = operation["replace"]
        expected = operation["expected_occurrences"]
        if search == before:
            raise ValueError("Search block equals the entire file; whole-file rewrite is forbidden")
        if len(search) > max(40000, int(len(before) * 0.80)):
            raise ValueError(f"Operation #{index} is too large for a surgical edit")
        count = updated.count(search)
        if count != expected:
            raise ValueError(f"Operation #{index}: expected {expected} occurrence(s), found {count} in {task['file']}")
        updated = updated.replace(search, replace, expected)
        operation_results.append({
            "index": index,
            "expected_occurrences": expected,
            "matched_occurrences": count,
            "search_sha256": sha256_bytes(search.encode("utf-8")),
            "replace_sha256": sha256_bytes(replace.encode("utf-8")),
        })

    if updated == before:
        raise ValueError("Editor produced no change")

    backup = backup_file(project_root, run_dir, target)
    atomic_write_text(target, updated)
    return {
        "action": "modify",
        "file": task["file"],
        "backup": backup,
        "operations": operation_results,
        "before_sha256": sha256_bytes(before.encode("utf-8")),
        "after_sha256": sha256_bytes(updated.encode("utf-8")),
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


# ==========================================
# 11. EDITOR NODE
# ==========================================
def editor_node(state: AgentState) -> AgentState:
    project_root = Path(state["project_address"])
    run_dir = Path(state["run_dir"])
    manifest = state["manifest"]
    stack = list(state.get("plan_stack", []))
    current_task = deepcopy(state.get("current_task")) if state.get("current_task") else None

    if current_task and current_task.get("review_status") == "PASS":
        current_task = None

    if current_task is None:
        if not stack:
            return {"editor_status": "EVALUATE", "workflow_status": "COMPLETED", "current_task": None, "issue_report": ""}
        # LIFO: exactly one task is removed from the top of the persisted stack.
        current_task = deepcopy(stack.pop())

    task_id = current_task["id"]
    target_file = current_task["file"]
    target_path = resolve_repo_path(project_root, target_file)
    target_manifest = manifest_target(manifest, target_file)
    target_language = target_manifest.get("language", infer_language(target_path))
    attempt_map = dict(state.get("task_attempts", {}))
    attempt = attempt_map.get(task_id, 0) + 1
    attempt_map[task_id] = attempt

    current_code = ""
    if target_path.exists():
        current_code = read_text_file(target_path)
    if len(current_code.encode("utf-8")) > MAX_FILE_BYTES:
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
    layered_context = render_retrieval_context(retrieval)

    lsp_context = ""
    if ENABLE_LSP and target_path.exists() and target_language in LANGUAGE_LSP_IDS:
        range_info = symbol_range(manifest, target_file, current_task.get("function", ""))
        line = max(0, (range_info[0] - 1) if range_info else 0)
        lsp_result = lsp_analyze_file(project_root, target_file, target_language, current_code, line=line, character=0)
        lsp_context = render_lsp_context(lsp_result)

    system_prompt = f"""
You are the Editor Agent in a production local repository-editing system.

LANGUAGE-SPECIFIC RULES:
{LANGUAGE_SYSTEM_PROMPTS.get(target_language, 'Use the repository language, toolchain, and existing conventions exactly.')}

GLOBAL RULES:
- Work only inside the provided repository.
- Preserve the architecture and unrelated code.
- Existing files MUST be edited by exact literal search/replace operations only.
- Every search string must be an exact substring copied from the CURRENT FILE.
- expected_occurrences must exactly match the number of literal matches.
- Never use regex, line-number replacements, placeholders, or full-file regeneration for an existing file.
- A whole-file replacement is forbidden even if it would be easier.
- New files may be returned as complete content only when the task explicitly says action=create.
- Do not perform unrelated refactors or formatting-only changes.
- Keep public APIs stable unless the task explicitly changes them.

CURRENT TARGET LANGUAGE: {target_language}

REPOSITORY CONTEXT:
{language_context(manifest, target_file)}

LSP CONTEXT:
{lsp_context or 'LSP context unavailable.'}
"""

    user_prompt = f"""
TARGET TASK:
{json.dumps(current_task, indent=2, ensure_ascii=False)}

OVERARCHING USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements']}

PREVIOUS REVIEW ISSUE:
{state.get('issue_report') or 'None'}

LAYERED REPOSITORY CONTEXT:
{layered_context}

CURRENT FILE:
{target_file}

CURRENT SOURCE:
{extract_target_context(current_code, target_manifest, current_task.get('function', '')) if len(current_code.encode('utf-8')) > MAX_LLM_FILE_BYTES else current_code}

Return ONLY JSON.
For an existing file:
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

For a new file:
{{
  "action": "create",
  "file": "{target_file}",
  "content": "COMPLETE NEW FILE CONTENT"
}}
"""
    write_log(run_dir, f"Editor Node [{task_id}] - SYSTEM PROMPT", system_prompt)
    write_log(run_dir, f"Editor Node [{task_id}] - USER PROMPT", user_prompt)

    started = time.perf_counter()
    raw = ""
    try:
        raw = invoke_llm(user_prompt, system_prompt=system_prompt)
        edit_plan = validate_edit_plan(safe_json_load(raw), current_task)
        result = apply_edit_plan(project_root, run_dir, current_task, edit_plan)
    except Exception as exc:
        duration = time.perf_counter() - started
        issue = (
            f"# Editor Issue — {task_id}\n\n"
            f"## Target\n`{target_file}` — `{current_task.get('function', '')}`\n\n"
            f"## Problem\n`{type(exc).__name__}: {exc}`\n\n"
            "## Required Action\nRegenerate a surgical JSON search/replace plan. Do not rewrite the file.\n"
        )
        atomic_write_text(run_dir / "issue.md", issue)
        write_log(run_dir, f"Editor Node [{task_id}] - APPLY ERROR", issue + f"\nRAW:\n{raw[:20000]}")
        return {
            "plan_stack": stack,
            "current_task": current_task,
            "issue_report": issue,
            "review_attempts": state.get("review_attempts", 0) + 1,
            "iterations": state.get("iterations", 0) + 1,
            "task_attempts": attempt_map,
            "editor_status": "RETRY",
            "workflow_status": "RUNNING",
            "task_durations": {
                **state.get("task_durations", {}),
                task_id: round(state.get("task_durations", {}).get(task_id, 0.0) + duration, 4),
            },
        }

    duration = time.perf_counter() - started
    file_attempts = dict(state.get("file_attempts", {}))
    file_attempts[target_file] = file_attempts.get(target_file, 0) + 1
    current_task["review_status"] = "PENDING"
    current_task["attempt"] = attempt
    current_task["last_edit_seconds"] = round(duration, 4)
    current_task["last_edit"] = result

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
    write_log(run_dir, f"Editor Node [{task_id}] - APPLIED EDIT", json.dumps(result, indent=2, ensure_ascii=False))
    return {
        "plan_stack": stack,
        "current_task": current_task,
        "issue_report": "",
        "review_attempts": 0,
        "iterations": state.get("iterations", 0) + 1,
        "file_attempts": file_attempts,
        "task_attempts": attempt_map,
        "task_durations": {
            **state.get("task_durations", {}),
            task_id: round(state.get("task_durations", {}).get(task_id, 0.0) + duration, 4),
        },
        "task_history": history,
        "last_edit": result,
        "editor_status": "REVIEW",
        "workflow_status": "RUNNING",
        "retrieval_context": layered_context,
        "lsp_context": lsp_context,
    }


# ==========================================
# 12. REVIEWER NODE
# ==========================================
def local_validation(project_root: Path, target_file: str, manifest: dict[str, Any]) -> tuple[bool, str]:
    target = resolve_repo_path(project_root, target_file)
    if not target.exists():
        return False, f"Target file does not exist: {target_file}"
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"Target file is not valid UTF-8: {target_file}"

    language = manifest_target(manifest, target_file).get("language", infer_language(target, text))
    if language == "Python":
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return False, f"Python syntax error: {exc}"
        ok, output = run_optional_command([shutil.which("python") or "python", "-m", "py_compile", str(target)], project_root, 30)
        if not ok and "No module named" not in output:
            return False, f"Python compile check failed: {output}"

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

    if target.suffix.lower() in {".ts", ".tsx"}:
        tsc = shutil.which("tsc")
        config = project_root / "tsconfig.json"
        if tsc and config.exists():
            # No emit; bounded check. This is a project-level signal, not a replacement for LSP.
            ok, output = run_optional_command([tsc, "--noEmit", "--pretty", "false", "--project", str(config)], project_root, 60)
            if not ok:
                return False, f"TypeScript project check failed: {output[:12000]}"

    if ENABLE_LSP and language in LANGUAGE_LSP_IDS:
        result = lsp_analyze_file(project_root, target_file, language, text)
        if result.get("available"):
            diagnostics = result.get("diagnostics", [])
            errors = [item for item in diagnostics if int(item.get("severity", 1)) == 1]
            if errors:
                return False, f"LSP diagnostics reported {len(errors)} error(s): {json.dumps(errors[:12], ensure_ascii=False)}"

    return True, "Local syntax/LSP validation passed."


def reviewer_node(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    project_root = Path(state["project_address"])
    manifest = state["manifest"]
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

    local_ok, local_feedback = local_validation(project_root, target_file, manifest)
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
    context = render_retrieval_context(retrieval)

    prompt = f"""
You are the Reviewer Agent for one surgical repository edit.

OVERARCHING USER REQUEST:
{state['user_request']}

REQUIREMENTS:
{state['requirements']}

MANIFEST:
{state['manifest_json'][:65000]}

TASK BEING REVIEWED:
{json.dumps(task, indent=2, ensure_ascii=False)}

TARGET FILE: {target_file}

LOCAL VALIDATION:
{local_feedback}

ACTUAL EDIT DIFF:
{diff}

CURRENT TARGET SOURCE:
{extract_target_context(target_code, manifest_target(manifest, target_file), task.get('function', ''))}

LAYERED CONTEXT:
{context}

Review only the requested task.
Check:
1. The exact target symbol/section was changed as requested.
2. The edit satisfies the task and acceptance criteria.
3. Language/framework/architecture consistency is preserved.
4. Imports, references, APIs, types, selectors, paths, and obvious runtime behavior remain valid.
5. No unrelated behavior was changed.
6. Local validation findings are resolved.

If fully correct, return ONLY:
PASS

Otherwise return ONLY JSON:
{{
  "status": "FAIL",
  "summary": "concise issue",
  "replace": "exact problematic source snippet",
  "with": "exact corrected source snippet",
  "reason": "specific task/manifest violation"
}}
"""
    write_log(run_dir, f"Reviewer Node [{task['id']}] - PROMPT", prompt)

    if not local_ok:
        review_output = json.dumps({
            "status": "FAIL",
            "summary": local_feedback,
            "replace": "",
            "with": "",
            "reason": local_feedback,
        }, ensure_ascii=False)
    else:
        review_output = invoke_llm(prompt)
    write_log(run_dir, f"Reviewer Node [{task['id']}] - OUTPUT", review_output)

    if re.match(r"^\s*PASS\.?\s*$", review_output, re.IGNORECASE):
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
        return {
            "current_task": approved,
            "issue_report": "",
            "review_attempts": 0,
            "completed_tasks": completed,
            "task_history": history,
            "workflow_status": "RUNNING",
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
    if status == "EVALUATE":
        return "evaluate"
    if status == "RETRY":
        if state.get("review_attempts", 0) >= MAX_REVIEW_ATTEMPTS:
            return "evaluate"
        return "retry"
    return "review"


def route_reviewer(state: AgentState) -> str:
    if not state.get("issue_report"):
        return "next"
    if state.get("review_attempts", 0) >= MAX_REVIEW_ATTEMPTS:
        return "evaluate"
    return "fix"


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
    tasks_completed = len(set(state.get("completed_tasks", [])))

    context = (
        f"USER REQUEST:\n{state['user_request']}\n\nREQUIREMENTS:\n{state['requirements']}\n\n"
        f"MANIFEST:\n{state['manifest_json'][:70000]}\n\nTASK HISTORY:\n"
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
        },
    )
    workflow.add_conditional_edges(
        "reviewer",
        route_reviewer,
        {
            "fix": "editor",
            "next": "editor",
            "evaluate": "evaluator",
        },
    )
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

    run_dir = Path.cwd() / "SDLC_Runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
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
        app.invoke(initial_state)
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
