"""
codebase_analyzer.py
=====================
Phase 1 add-on for loop_workflow_main.py: ingest a PRE-BUILT project
(monolithic single script OR multi-file / multi-folder repo) and turn it
into a manifest an LLM can navigate cheaply.

Two problems this solves:

  1. A file can be 4000+ lines -> can't hand the whole thing to the LLM.
     Fix: split every file into logical chunks (functions/classes for
     Python via `ast`, regex-detected blocks for JS/TS, sliding windows
     as the last-resort fallback for anything else), and summarize each
     chunk with ONE small LLM call. Raw code for chunk N is never mixed
     into the summarization call for chunk N+1 -> bounded context per call
     regardless of file size.

  2. The editor needs "context for a particular part" without re-reading
     everything. Fix: the manifest stores name / signature / purpose /
     line-range per symbol (cheap - no source code inside the manifest
     itself). At edit time, hand the LLM the symbol table first (a menu),
     let it name the symbol/lines it needs, then read EXACTLY those lines
     straight off disk. The manifest is the index; disk is the source of
     truth, so it can never go stale/hallucinated.

Drop this file next to loop_workflow_main.py and import what you need
(see the integration notes + `analyze_existing_project_node` at the
bottom for wiring it into the existing StateGraph).
"""

import ast
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ==========================================
# CONFIG
# ==========================================
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv", "env",
    "dist", "build", ".idea", ".vscode", "SDLC_Runs", ".pytest_cache",
}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
                    ".java", ".go", ".rb", ".php"}

MAX_CHUNK_LINES = 200            # hard ceiling per chunk sent to the LLM
CHUNK_OVERLAP_LINES = 15         # only used by sliding-window sub-splitting
MAX_CHUNK_CHARS_FOR_LLM = 6000   # belt-and-suspenders alongside the line cap


# ==========================================
# DATA MODEL
# ==========================================
@dataclass
class CodeChunk:
    name: str
    kind: str                    # "function" | "class" | "method" | "block"
    parent: Optional[str]
    line_start: int
    line_end: int
    source: str


@dataclass
class SymbolSummary:
    name: str
    kind: str
    parent: Optional[str]
    line_start: int
    line_end: int
    signature: str
    purpose: str
    calls: list
    dom_ids: list


@dataclass
class FileManifest:
    path: str
    language: str
    total_lines: int
    imports: list
    file_summary: str
    symbols: list  # list[dict] (SymbolSummary.__dict__)


# ==========================================
# 1. DISCOVERY
# ==========================================
def discover_files(root: Path) -> list:
    """Walks `root` (or returns it directly if it's a single monolithic
    script) and returns every code file, skipping noise directories."""
    root = Path(root)
    if root.is_file():
        return [root]

    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in CODE_EXTENSIONS:
                files.append(p)
    return sorted(files)


def _guess_language(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript", ".html": "html",
        ".css": "css", ".java": "java", ".go": "go", ".rb": "ruby", ".php": "php",
    }.get(path.suffix.lower(), "unknown")


# ==========================================
# 2. CHUNKING
# ==========================================
def _split_if_too_big(name, kind, parent, start, end, src) -> list:
    """A single function/class can itself be huge (e.g. one 1200-line
    handler). If so, sub-split it into overlapping windows but keep the
    logical name attached to every part, so the manifest still reads as
    one symbol with a 'part N' suffix rather than losing the boundary."""
    if end - start <= MAX_CHUNK_LINES and len(src) <= MAX_CHUNK_CHARS_FOR_LLM:
        return [CodeChunk(name, kind, parent, start, end, src)]

    sub_lines = src.splitlines()
    out, i, part = [], 0, 1
    while i < len(sub_lines):
        window = sub_lines[i:i + MAX_CHUNK_LINES]
        out.append(CodeChunk(
            f"{name} (part {part})", kind, parent,
            start + i, start + i + len(window) - 1, "\n".join(window),
        ))
        i += MAX_CHUNK_LINES - CHUNK_OVERLAP_LINES
        part += 1
    return out


def chunk_python_file(path: Path):
    """AST-based chunking: one chunk per top-level function, one per
    method (linked to its class via `parent`), one small header chunk
    per class, one chunk per top-level statement run. Falls back to
    generic chunking on a SyntaxError (e.g. Python 2 files)."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_generic_file(path), []

    imports, chunks = [], []

    def _segment(node) -> str:
        seg = ast.get_source_segment(source, node)
        return seg if seg is not None else "\n".join(lines[node.lineno - 1:node.end_lineno])

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node) if hasattr(ast, "unparse") else _segment(node))
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(_split_if_too_big(
                node.name, "function", None, node.lineno, node.end_lineno, _segment(node)))

        elif isinstance(node, ast.ClassDef):
            method_nodes = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if method_nodes:
                first_method_line = method_nodes[0].lineno
                header_src = "\n".join(lines[node.lineno - 1:first_method_line - 1])
                chunks.append(CodeChunk(node.name, "class", None, node.lineno, first_method_line - 1, header_src))
                for m in method_nodes:
                    chunks.extend(_split_if_too_big(
                        m.name, "method", node.name, m.lineno, m.end_lineno, _segment(m)))
            else:
                chunks.append(CodeChunk(node.name, "class", None, node.lineno, node.end_lineno, _segment(node)))

        else:
            chunks.extend(_split_if_too_big(
                f"module_level_L{node.lineno}", "block", None, node.lineno, node.end_lineno, _segment(node)))

    return chunks, imports


_JS_BOUNDARY = re.compile(
    r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+(\w+)|'
    r'^\s*(export\s+)?const\s+(\w+)\s*=\s*(async\s*)?\(.*?\)\s*=>|'
    r'^\s*(export\s+)?class\s+(\w+)',
    re.MULTILINE,
)


def chunk_generic_file(path: Path) -> list:
    """Non-Python chunking. Tries a regex function/class boundary scan for
    JS/TS (good enough to keep related logic together); everything else
    (HTML, CSS, or JS/TS with no matched boundaries) falls back to a
    plain sliding window, which always terminates and always respects
    the size ceiling."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    ext = path.suffix.lower()

    if ext in {".js", ".ts", ".jsx", ".tsx"}:
        offsets = [m.start() for m in _JS_BOUNDARY.finditer(source)]
        if offsets:
            return _chunks_from_char_boundaries(source, lines, offsets)

    return _sliding_window_chunks(lines)


def _chunks_from_char_boundaries(source, lines, char_offsets) -> list:
    line_starts = [source[:o].count("\n") + 1 for o in char_offsets] + [len(lines) + 1]
    chunks = []
    for idx in range(len(line_starts) - 1):
        start, end = line_starts[idx], line_starts[idx + 1] - 1
        if end < start:
            continue
        snippet = "\n".join(lines[start - 1:end])
        name_match = re.search(r'\b(?:function|class|const)\s+(\w+)', snippet)
        name = name_match.group(1) if name_match else f"block_L{start}"
        chunks.extend(_split_if_too_big(name, "block", None, start, end, snippet))
    return chunks


def _sliding_window_chunks(lines) -> list:
    chunks, i, part = [], 0, 1
    while i < len(lines):
        window = lines[i:i + MAX_CHUNK_LINES]
        chunks.append(CodeChunk(
            f"block_part_{part}", "block", None, i + 1, i + len(window), "\n".join(window)))
        i += MAX_CHUNK_LINES - CHUNK_OVERLAP_LINES
        part += 1
    return chunks


# ==========================================
# 3. PER-CHUNK SUMMARIZATION (map step)
# ==========================================
_SUMMARY_PROMPT = """You are indexing a codebase so another LLM can find and edit the right \
piece of code later WITHOUT reading the whole file.

FILE: {file_path} ({language})
CHUNK: {name}  (source lines {start}-{end})

CODE:
{code}

Return ONLY a JSON object, no markdown fences, no explanation, with exactly these keys:
{{
  "signature": "the function/class signature or a short one-line identifier",
  "purpose": "1-2 sentence plain-English summary of what this code does",
  "calls": ["other function or symbol names this code calls, if any"],
  "dom_ids": ["any HTML element IDs or CSS classes this code reads or writes, if any"]
}}
"""


def _safe_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def summarize_chunk(llm, file_path: str, language: str, chunk: CodeChunk) -> SymbolSummary:
    prompt = _SUMMARY_PROMPT.format(
        file_path=file_path, language=language, name=chunk.name,
        start=chunk.line_start, end=chunk.line_end,
        code=chunk.source[:MAX_CHUNK_CHARS_FOR_LLM],
    )
    try:
        raw = llm.invoke(prompt).content.strip()
    except Exception as e:
        raw = ""
        print(f"   [!] summarization failed for {chunk.name}: {e}")
    data = _safe_json(raw)
    return SymbolSummary(
        name=chunk.name, kind=chunk.kind, parent=chunk.parent,
        line_start=chunk.line_start, line_end=chunk.line_end,
        signature=data.get("signature", chunk.name),
        purpose=data.get("purpose", "(summary unavailable - review manually)"),
        calls=data.get("calls") or [],
        dom_ids=data.get("dom_ids") or [],
    )


# ==========================================
# 4. FILE-LEVEL REDUCE
# ==========================================
_FILE_SUMMARY_PROMPT = """Summarize this file's overall responsibility in 2-3 sentences, \
based only on the symbol list below (not raw code). Plain text, no JSON, no preamble.

FILE: {file_path}
SYMBOLS:
{symbol_lines}
"""


def build_file_manifest(llm, path: Path, scan_root: Path) -> FileManifest:
    language = _guess_language(path)
    if language == "python":
        chunks, imports = chunk_python_file(path)
    else:
        chunks, imports = chunk_generic_file(path), []

    rel_path = str(path.relative_to(scan_root))
    symbols = [summarize_chunk(llm, rel_path, language, c) for c in chunks]

    symbol_lines = "\n".join(f"- {s.kind} {s.name}: {s.purpose}" for s in symbols) or "(no symbols found)"
    try:
        file_summary = llm.invoke(
            _FILE_SUMMARY_PROMPT.format(file_path=rel_path, symbol_lines=symbol_lines)
        ).content.strip()
    except Exception as e:
        file_summary = f"(summary unavailable: {e})"

    return FileManifest(
        path=rel_path,
        language=language,
        total_lines=len(path.read_text(encoding="utf-8", errors="ignore").splitlines()),
        imports=imports,
        file_summary=file_summary,
        symbols=[asdict(s) for s in symbols],
    )


# ==========================================
# 5. PROJECT-LEVEL ASSEMBLY
# ==========================================
def build_project_manifest(llm, source_path: str, run_dir: Path) -> dict:
    root = Path(source_path)
    scan_root = root if root.is_dir() else root.parent
    files = discover_files(root)

    file_manifests = []
    for f in files:
        print(f"   \U0001F4C4 Indexing {f.relative_to(scan_root)} ...")
        file_manifests.append(asdict(build_file_manifest(llm, f, scan_root)))

    manifest = {
        "root": str(scan_root),
        "project_type": "monolithic" if len(files) == 1 else "multi_file",
        "file_count": len(files),
        "files": file_manifests,
    }

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "project_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "PROJECT_MAP.md").write_text(_render_markdown_map(manifest), encoding="utf-8")
    return manifest


def _render_markdown_map(manifest: dict) -> str:
    lines = [f"# Project Map \u2014 {manifest['root']}",
             f"_{manifest['file_count']} file(s), {manifest['project_type']}_\n"]
    for fm in manifest["files"]:
        lines.append(f"## {fm['path']}  ({fm['language']}, {fm['total_lines']} lines)")
        lines.append(fm["file_summary"] + "\n")
        for s in fm["symbols"]:
            loc = f"L{s['line_start']}-{s['line_end']}"
            owner = f"{s['parent']}." if s.get("parent") else ""
            lines.append(f"- `{owner}{s['name']}` [{s['kind']}, {loc}] \u2014 {s['purpose']}")
        lines.append("")
    return "\n".join(lines)


# ==========================================
# 6. ON-DEMAND RETRIEVAL (used by the editor loop, Phase 2)
# ==========================================
def get_symbol_table(manifest: dict, file_path: str) -> list:
    """Cheap 'menu': names + signatures + line ranges only, no source code.
    Hand this to the editor LLM first so it can pick what it actually needs."""
    for fm in manifest["files"]:
        if fm["path"] == file_path:
            return [{"name": s["name"], "kind": s["kind"], "signature": s["signature"],
                      "lines": [s["line_start"], s["line_end"]]} for s in fm["symbols"]]
    return []


def get_exact_source(root: str, file_path: str, line_start: int, line_end: int, context: int = 3) -> str:
    """Reads ONLY the requested lines (plus a little surrounding context)
    straight off disk. The manifest never stores source, so this can
    never return stale or hallucinated code."""
    full_path = Path(root) / file_path
    lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lo = max(0, line_start - 1 - context)
    hi = min(len(lines), line_end + context)
    return "\n".join(lines[lo:hi])


# ==========================================
# 7. LANGGRAPH INTEGRATION NODE
# ==========================================
def analyze_existing_project_node(state):
    """
    Phase-1 entry point for a PRE-BUILT project (monolithic or multi-file).

    Expects on `state`:
        source_dir : str  -> path to the existing project (file or folder)
        run_dir     : Path -> same run_dir the rest of the graph already uses

    Wiring it into loop_workflow_main.py's existing StateGraph:

        from codebase_analyzer import analyze_existing_project_node, llm as analyzer_llm
        workflow.add_node("analyze_existing", analyze_existing_project_node)

        # route to it instead of "architect" when the user supplied an
        # existing project, e.g. via a conditional entry point:
        workflow.set_conditional_entry_point(
            lambda s: "analyze_existing" if s.get("source_dir") else "requirements",
            {"analyze_existing": "analyze_existing", "requirements": "requirements"},
        )
        workflow.add_edge("analyze_existing", "editor")

    Note this returns `pending_files` as the list of existing file paths
    (relative to source_dir), not the from-scratch template used by
    architect_node. The editor_node would need a small edit for this
    mode: instead of pulling full content from state["file_system"], call
    get_symbol_table(...) + get_exact_source(...) to pull just the slice
    it's working on. Happy to make that edit to editor_node/reviewer_node
    directly if you want the full loop wired end-to-end.
    """
    from datetime import datetime

    print("\U0001F5C2\uFE0F  [Analyzer] Scanning existing project...")
    manifest = build_project_manifest(llm, state["source_dir"], state["run_dir"])

    log_file = Path(state["run_dir"]) / "execution_log.txt"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%H:%M:%S')}] === ANALYZER NODE ===\n")
        f.write(f"{manifest['file_count']} files indexed as a {manifest['project_type']} project.\n")
        f.write("-" * 80 + "\n")

    return {
        "project_manifest": json.dumps(manifest),
        "pending_files": [fm["path"] for fm in manifest["files"]],
        "review_attempts": 0,
        "feedback": "",
    }


# ==========================================
# 8. STANDALONE LLM HANDLE (swap for your own ChatOllama instance,
#    or pass a different `llm` into the functions above directly)
# ==========================================
llm = None
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="gemma4:e2b", num_predict=1024, num_ctx=8192, temperature=0.1)
except Exception:
    pass  # analyzer functions all accept `llm` as a parameter, so this is optional


# ==========================================
# 9. STANDALONE TEST
# ==========================================
if __name__ == "__main__":
    import sys

    if llm is None:
        print("No default LLM configured - edit section 8 or import build_project_manifest "
              "and pass your own `llm` object.")
        sys.exit(1)

    target = input("Path to existing project (file or folder): ").strip()
    out_dir = Path.cwd() / "SDLC_Runs" / "manifest_only_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = build_project_manifest(llm, target, out_dir)
    print(f"\n\u2705 Indexed {result['file_count']} file(s). "
          f"See {out_dir/'project_manifest.json'} and {out_dir/'PROJECT_MAP.md'}")
