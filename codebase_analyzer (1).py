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

Known limitations (documented rather than silently wrong):
  - JS/TS chunking is regex-based, not a real parser. It catches top-level
    function/const-arrow/class declarations but not deeply nested closures.
    Swap in tree-sitter if you need grammar-accurate boundaries.
  - Python nested classes (a class defined inside another class/function)
    are captured as one block under the outer scope rather than being
    individually indexed - single-level `parent` linkage only.

Drop this file next to loop_workflow_main.py and import what you need
(see the integration notes + `analyze_existing_project_node` at the
bottom for wiring it into the existing StateGraph).
"""

import ast
import hashlib
import json
import os
import textwrap
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

MAX_CHUNK_LINES = 200             # hard ceiling per chunk sent to the LLM
CHUNK_OVERLAP_LINES = 15          # only used by sliding-window sub-splitting
MAX_CHUNK_CHARS_FOR_LLM = 6000    # belt-and-suspenders alongside the line cap
MAX_CHUNKS_PER_FILE = 150         # bounds worst-case LLM calls on one giant/generated file


# ==========================================
# DATA MODEL
# ==========================================
@dataclass
class CodeChunk:
    name: str
    kind: str                     # "function" | "class" | "method" | "block"
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
    call_reasons: dict
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
    if not root.exists():
        raise FileNotFoundError(f"source path does not exist: {root}")
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


def _detect_entry_points(files: list, scan_root: Path) -> list:
    """Heuristic entry-point detection surfaced at the top of the manifest,
    so the LLM (or you) can immediately see where execution starts without
    hunting through every file's symbol table."""
    common_names = {"index.html", "main.py", "app.py", "server.js",
                     "index.js", "main.js", "manage.py"}
    entries = []
    for f in files:
        if f.name in common_names:
            entries.append(str(f.relative_to(scan_root)))
        elif f.suffix == ".py":
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', text):
                rel = str(f.relative_to(scan_root))
                if rel not in entries:
                    entries.append(rel)
    return entries


# ==========================================
# 2. CHUNKING
# ==========================================
def _split_if_too_big(name, kind, parent, start, end, src) -> list:
    """A single function/class can itself be huge (e.g. one 1200-line
    handler, or one 20,000-char generated/minified line). If so, sub-split
    it into windows that respect BOTH the line count and char-count
    ceilings - checked before a line is added, not after, so one
    pathological line can never blow a chunk past the size cap - and keep
    the logical name attached to every part rather than losing the
    boundary."""
    if end - start <= MAX_CHUNK_LINES and len(src) <= MAX_CHUNK_CHARS_FOR_LLM:
        return [CodeChunk(name, kind, parent, start, end, src)]

    sub_lines = src.splitlines()
    out, i, part, n = [], 0, 1, len(sub_lines)
    while i < n:
        line = sub_lines[i]
        if len(line) > MAX_CHUNK_CHARS_FOR_LLM:
            # one line alone exceeds the cap (e.g. a minified bundle) - hard-slice it
            out.append(CodeChunk(f"{name} (part {part})", kind, parent,
                                  start + i, start + i, line[:MAX_CHUNK_CHARS_FOR_LLM]))
            i += 1
            part += 1
            continue

        window, char_count, j = [], 0, i
        while j < n and (j - i) < MAX_CHUNK_LINES and char_count + len(sub_lines[j]) + 1 <= MAX_CHUNK_CHARS_FOR_LLM:
            window.append(sub_lines[j])
            char_count += len(sub_lines[j]) + 1
            j += 1
        if not window:
            window = [sub_lines[i][:MAX_CHUNK_CHARS_FOR_LLM]]
            j = i + 1

        out.append(CodeChunk(f"{name} (part {part})", kind, parent,
                              start + i, start + j - 1, "\n".join(window)))
        i += max(1, (j - i) - CHUNK_OVERLAP_LINES)
        part += 1
    return out


def _decorated_span(node):
    """Since Python 3.8, `node.lineno` for a decorated function/class
    starts at the `def`/`class` keyword - NOT the decorator line. Left
    unfixed, `@app.route(...)`, `@staticmethod`, `@dataclass`, etc. would
    silently vanish from every chunk (and from the editor's exact-line
    retrieval later). This expands the start line to cover them."""
    start = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = min(start, min(d.lineno for d in decorators))
    return start, node.end_lineno


def chunk_python_file(path: Path):
    """AST-based chunking: one chunk per top-level function (decorators
    included), one per method (linked to its class via `parent`), one
    header chunk per class, and consecutive top-level non-def/class
    statements (constants, config, argparse setup, ...) are grouped into
    a single chunk instead of one LLM call per statement. Falls back to
    generic chunking on a parse error (e.g. Python 2 files)."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    if not source.strip():
        return [], []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return chunk_generic_file(path), []

    def _slice(start, end) -> str:
        return "\n".join(lines[start - 1:end])

    imports, chunks, pending = [], [], []

    def flush_pending():
        if not pending:
            return
        start, end = pending[0].lineno, pending[-1].end_lineno
        name = f"module_level_L{start}" if len(pending) == 1 else f"module_level_L{start}-{end}"
        chunks.extend(_split_if_too_big(name, "block", None, start, end, _slice(start, end)))
        pending.clear()

    def emit_class(node):
        class_start, class_end = _decorated_span(node)
        method_nodes = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        if not method_nodes:
            chunks.append(CodeChunk(node.name, "class", None, class_start, class_end,
                                     _slice(class_start, class_end)))
            return

        # header = decorators + docstring + class-level attrs before the first method
        first_start, _ = _decorated_span(method_nodes[0])
        if first_start - 1 >= class_start:
            chunks.append(CodeChunk(node.name, "class", None, class_start, first_start - 1,
                                     _slice(class_start, first_start - 1)))

        for idx, m in enumerate(method_nodes):
            m_start, m_end = _decorated_span(m)
            chunks.extend(_split_if_too_big(m.name, "method", node.name, m_start, m_end,
                                             _slice(m_start, m_end)))
            # capture anything sitting between this method and the next (class
            # attrs interleaved with methods, nested Meta classes, etc.) so no
            # source line is ever silently dropped from the index
            next_start = _decorated_span(method_nodes[idx + 1])[0] if idx + 1 < len(method_nodes) else class_end + 1
            gap_start = m_end + 1
            if gap_start < next_start:
                gap_src = _slice(gap_start, next_start - 1)
                if gap_src.strip():
                    chunks.append(CodeChunk(f"{node.name}._body_L{gap_start}", "block", node.name,
                                             gap_start, next_start - 1, gap_src))

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            flush_pending()
            imports.append(ast.unparse(node) if hasattr(ast, "unparse")
                            else _slice(node.lineno, node.end_lineno))
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_pending()
            start, end = _decorated_span(node)
            chunks.extend(_split_if_too_big(node.name, "function", None, start, end, _slice(start, end)))

        elif isinstance(node, ast.ClassDef):
            flush_pending()
            emit_class(node)

        else:
            pending.append(node)

    flush_pending()
    return chunks, imports


_JS_BOUNDARY = re.compile(
    r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s*\*?\s+(\w+)|'
    r'^\s*(export\s+)?const\s+(\w+)\s*=\s*(async\s*)?\(.*?\)\s*=>|'
    r'^\s*(export\s+)?class\s+(\w+)',
    re.MULTILINE,
)



def chunk_generic_file(path: Path) -> list:
    """Non-Python chunking. Tries a regex function/class boundary scan for
    JS/TS (good enough to keep related logic together); everything else
    (HTML, CSS, or JS/TS with no matched boundaries) falls back to a
    sliding window that is both line- AND char-bounded, so a single
    pathological line (e.g. one minified/generated line) can't blow past
    the per-chunk size ceiling either."""
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    if not source.strip():
        return []
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
        name_match = re.search(r'\b(?:function\*?|class|const)\s+(\w+)', snippet)
        name = name_match.group(1) if name_match else f"block_L{start}"
        chunks.extend(_split_if_too_big(name, "block", None, start, end, snippet))
    return chunks


def _sliding_window_chunks(lines) -> list:
    """Fallback for files with no detected function/class boundaries at
    all (plain HTML/CSS, or JS/TS the regex scan didn't match). Delegates
    to the same line+char bounded splitter used everywhere else, so a
    single pathological line (minified/generated code) is handled
    identically instead of duplicating that logic."""
    if not lines:
        return []
    return _split_if_too_big("block", "block", None, 1, len(lines), "\n".join(lines))


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
  "calls": ["other function or symbol names this code calls, if any - else []"],
  "call_reasons": {{"call_name": "short phrase: why this code calls it"}},
  "dom_ids": ["HTML element IDs or CSS classes this reads/writes, if any - else []"]
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

_CALL_REGEX = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
_PY_NOISE = {
    "if", "for", "while", "with", "def", "class", "return", "elif", "else",
    "try", "except", "finally", "lambda", "yield", "assert", "raise",
    "and", "or", "not", "in", "is", "print", "super", "self",
}

def _extract_calls_regex(src: str) -> list:
    """Fallback call extraction for non-Python chunks / broken Python
    sub-splits: regex scan for `name(`, minus control-flow keywords."""
    return sorted({m for m in _CALL_REGEX.findall(src) if m.lower() not in _PY_NOISE})


def _extract_calls_ast(src: str) -> list:
    """Deterministic call extraction so CALLS edges never depend solely
    on the summarizer LLM. Bare names only (self.move()/ball.move() ->
    'move') to line up with the symbol table, which is keyed by bare name."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except (SyntaxError, ValueError):
        return _extract_calls_regex(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return sorted(n for n in names if n.lower() not in _PY_NOISE)

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

    # for call issue fix
    deterministic = (_extract_calls_ast(chunk.source) if language == "python"
                     else _extract_calls_regex(chunk.source))
    merged_calls = sorted(set(data.get("calls") or []) | set(deterministic))
    call_reasons = {k: v for k, v in (data.get("call_reasons") or {}).items() if k in merged_calls}

    return SymbolSummary(
        name=chunk.name, kind=chunk.kind, parent=chunk.parent,
        line_start=chunk.line_start, line_end=chunk.line_end,
        signature=data.get("signature", chunk.name),
        purpose=data.get("purpose", "(summary unavailable - review manually)"),
        # was: calls=data.get("calls") or [],
        calls=merged_calls,
        call_reasons=call_reasons,
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
    rel_path = str(path.relative_to(scan_root))
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    total_lines = len(raw_text.splitlines())

    if not raw_text.strip():
        return FileManifest(path=rel_path, language=language, total_lines=total_lines,
                             imports=[], file_summary="(empty file)", symbols=[])

    if language == "python":
        chunks, imports = chunk_python_file(path)
    else:
        chunks, imports = chunk_generic_file(path), []

    truncated = len(chunks) > MAX_CHUNKS_PER_FILE
    if truncated:
        print(f"   [!] {rel_path}: {len(chunks)} chunks found, capping at {MAX_CHUNKS_PER_FILE} "
              f"(large generated/vendored file? consider excluding it)")
        chunks = chunks[:MAX_CHUNKS_PER_FILE]

    symbols = [summarize_chunk(llm, rel_path, language, c) for c in chunks]

    symbol_lines = "\n".join(f"- {s.kind} {s.name}: {s.purpose}" for s in symbols) or "(no symbols found)"
    try:
        file_summary = llm.invoke(
            _FILE_SUMMARY_PROMPT.format(file_path=rel_path, symbol_lines=symbol_lines)
        ).content.strip()
    except Exception as e:
        file_summary = f"(summary unavailable: {e})"
    if truncated:
        file_summary += "\n\n[NOTE: this file exceeded the chunk cap - only part of it was indexed.]"

    return FileManifest(
        path=rel_path, language=language, total_lines=total_lines,
        imports=imports, file_summary=file_summary, symbols=[asdict(s) for s in symbols],
    )


# ==========================================
# 5. PROJECT-LEVEL ASSEMBLY (with incremental, resumable caching)
# ==========================================
def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache_path: Path, cache: dict):
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def build_project_manifest(llm, source_path: str, run_dir: Path, use_cache: bool = True) -> dict:
    """Builds (or incrementally refreshes) the project manifest.

    With use_cache=True (default), each file's manifest is cached by
    content hash in run_dir/manifest_cache.json and the cache is saved
    after EVERY file, not just at the end - so a rerun after editing one
    file only re-indexes that file (fast, and cheap on LLM calls), and a
    crash partway through a big project doesn't lose the work already done.
    """
    root = Path(source_path)
    scan_root = root if root.is_dir() else root.parent
    files = discover_files(root)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "manifest_cache.json"
    cache = _load_cache(cache_path) if use_cache else {}

    file_manifests = []
    for f in files:
        rel = str(f.relative_to(scan_root))
        digest = _file_hash(f)
        cached = cache.get(rel)

        if use_cache and cached and cached.get("hash") == digest:
            print(f"   \u2713 cached   {rel} (unchanged)")
            file_manifests.append(cached["manifest"])
            continue

        print(f"   \U0001F4C4 indexing  {rel} ...")
        fm = asdict(build_file_manifest(llm, f, scan_root))
        file_manifests.append(fm)
        if use_cache:
            cache[rel] = {"hash": digest, "manifest": fm}
            _save_cache(cache_path, cache)  # persist after every file - crash-safe

    if use_cache:
        live_paths = {str(f.relative_to(scan_root)) for f in files}
        cache = {k: v for k, v in cache.items() if k in live_paths}  # drop deleted files
        _save_cache(cache_path, cache)

    manifest = {
        "root": str(scan_root),
        "project_type": "monolithic" if len(files) == 1 else "multi_file",
        "file_count": len(files),
        "entry_points": _detect_entry_points(files, scan_root),
        "files": file_manifests,
    }

    (run_dir / "project_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "PROJECT_MAP.md").write_text(_render_markdown_map(manifest), encoding="utf-8")
    return manifest


def _render_markdown_map(manifest: dict) -> str:
    lines = [f"# Project Map \u2014 {manifest['root']}",
             f"_{manifest['file_count']} file(s), {manifest['project_type']}_"]
    if manifest.get("entry_points"):
        lines.append(f"_Entry points: {', '.join(manifest['entry_points'])}_")
    lines.append("")
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


def find_symbol(manifest: dict, name: str) -> list:
    """Search every file's symbol table for a name match (exact or
    substring, case-insensitive) so the editor can locate 'the login
    function' without already knowing which file it lives in."""
    needle = name.lower()
    hits = []
    for fm in manifest["files"]:
        for s in fm["symbols"]:
            if needle == s["name"].lower() or needle in s["name"].lower():
                hits.append({"file": fm["path"], "name": s["name"], "kind": s["kind"],
                             "signature": s["signature"], "lines": [s["line_start"], s["line_end"]]})
    return hits


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
    get_symbol_table(...) / find_symbol(...) + get_exact_source(...) to
    pull just the slice it's working on. Happy to make that edit to
    editor_node/reviewer_node directly if you want the full loop wired
    end-to-end.
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
except Exception as e:
    print(f"[codebase_analyzer] no default LLM configured ({e}); "
          f"pass your own `llm` into build_project_manifest(...) instead.")


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
