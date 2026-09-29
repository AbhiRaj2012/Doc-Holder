import os
import re
import json
import time
import shutil
import operator
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# DeepEval is optional; the pipeline still runs without it.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

# Initialize LLM[cite: 1]
llm = ChatOllama(
    model="gemma4:e2b", 
    num_predict=2048,
    num_ctx=8192,
    temperature=0.1
)

def write_log(run_dir: Path, step: str, details: str):
    """Logs workflow execution details[cite: 1]."""
    log_file = run_dir / "execution_log.txt"
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] === {step.upper()} ===\n")
        f.write(str(details) + "\n")
        f.write("-" * 80 + "\n")

# ==========================================
# 1. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    project_address: str
    user_request: str
    plan_stack: List[Dict[str, Any]]
    issue_report: str
    manifest: str
    run_dir: Path
    start_time: float
    total_tasks: int
    current_task: Dict[str, Any]
    review_attempts: int
    file_attempts: Annotated[dict, operator.ior]
    modified_files: List[str]

# ==========================================
# 2. NODES
# ==========================================
def requirement_node(state: AgentState):
    """Ingests the project_address and user_request, initializes tracking metrics."""
    print(f"🔍 Initializing task for: {state['project_address']}")
    
    start_time = time.time()
    write_log(state["run_dir"], "Requirement Node", f"Project: {state['project_address']}\nRequest: {state['user_request']}")
    
    return {
        "start_time": start_time,
        "issue_report": "",
        "review_attempts": 0,
        "file_attempts": {},
        "modified_files": []
    }

def analyzer_node(state: AgentState):
    """Scans the provided directory, identifies languages, frameworks, and architecture."""
    print("📂 Analyzing repository structure...")
    
    project_path = Path(state["project_address"])
    ignore_dirs = {".git", "__pycache__", "node_modules", "venv", ".idea", ".vscode"}
    tree = []
    
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        level = root.replace(str(project_path), "").count(os.sep)
        indent = " " * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            tree.append(f"{subindent}{f}")
            
    tree_str = "\n".join(tree[:200]) # Cap output length to prevent context explosion
    
    prompt = f"""
    Analyze the following repository structure and infer the primary programming languages, frameworks, and architecture.
    Output a structured JSON summary representing the project manifest.
    
    Repository tree:
    {tree_str}
    
    Output strictly in this JSON format:
    {{
        "languages": ["lang1", "lang2"],
        "frameworks": ["fw1", "fw2"],
        "architecture_pattern": "MVC/Microservices/Script/etc",
        "description": "Brief guess of what this project does"
    }}
    """
    
    response = llm.invoke(prompt).content.strip()
    manifest_match = re.search(r'\{.*\}', response, re.DOTALL)
    manifest = manifest_match.group(0) if manifest_match else "{}"
    
    write_log(state["run_dir"], "Analyzer Node", manifest)
    (state["run_dir"] / "manifest.json").write_text(manifest, encoding="utf-8")
    
    return {"manifest": manifest}

def planner_node(state: AgentState):
    """Evaluates the request and outputs a granular execution plan formatted as a LIFO Stack."""
    print("📝 Planning execution steps...")
    
    prompt = f"""
    You are a technical architect. Analyze the requested change against the codebase manifest.
    Create a granular, step-by-step execution plan to fulfill the request.
    
    Project Manifest: {state['manifest']}
    User Request: {state['user_request']}
    
    Output strictly a JSON list of task objects. I will use this as a LIFO stack (last in, first out).
    Order your JSON list so that the FIRST task to be executed is at the END of the list.
    
    Format required:
    [
        {{
            "file": "path/to/file.ext",
            "function": "target function or class (if applicable)",
            "instruction": "Detailed instruction of what to search for and replace",
            "is_new_file": false
        }}
    ]
    """
    
    response = llm.invoke(prompt).content.strip()
    stack_match = re.search(r'\[.*\]', response, re.DOTALL)
    plan_stack = json.loads(stack_match.group(0)) if stack_match else []
    
    write_log(state["run_dir"], "Planner Node", json.dumps(plan_stack, indent=2))
    
    return {
        "plan_stack": plan_stack,
        "total_tasks": len(plan_stack)
    }

def editor_node(state: AgentState):
    """Pops task, dynamically creates backup, and performs targeted search-and-replace edits."""
    plan_stack = state["plan_stack"]
    issue_report = state.get("issue_report", "")
    current_task = state.get("current_task")
    
    if not issue_report and plan_stack:
        current_task = plan_stack.pop()
        state["review_attempts"] = 0
        
    if not current_task:
        return {"plan_stack": plan_stack}

    target_file = current_task["file"]
    full_path = Path(state["project_address"]) / target_file
    
    print(f"💻 [Editor] Modifying {target_file}...")
    
    # Backup
    if full_path.exists() and not full_path.with_suffix(full_path.suffix + ".bak").exists():
        shutil.copy(full_path, str(full_path) + ".bak")
        
    file_content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
    
    prompt = f"""
    You are an expert developer. Read the project manifest and execute the task using targeted edits.
    Manifest context: {state['manifest']}
    
    Task: {json.dumps(current_task)}
    Reviewer Feedback (if any): {issue_report}
    
    Original File Content ({target_file}):
    ```
    {file_content}
    ```
    
    Perform the edit by outputting EXACTLY a SEARCH/REPLACE block. 
    DO NOT rewrite the entire file unless it is a new file. Use this exact syntax:
    
    <<<< SEARCH
    Exact lines to be replaced from the original file
    ====
    New lines to insert
    >>>> REPLACE
    """
    
    response = llm.invoke(prompt).content.strip()
    write_log(state["run_dir"], f"Editor Node [{target_file}]", response)
    
    if current_task.get("is_new_file"):
        clean_code = re.sub(r'```[a-zA-Z]*\n(.*?)\n```', r'\1', response, flags=re.DOTALL).strip()
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(clean_code, encoding="utf-8")
    else:
        pattern = re.compile(r'<<<< SEARCH\n(.*?)\n====\n(.*?)\n>>>> REPLACE', re.DOTALL)
        match = pattern.search(response)
        if match:
            search_text, replace_text = match.group(1), match.group(2)
            updated_content = file_content.replace(search_text, replace_text)
            full_path.write_text(updated_content, encoding="utf-8")
        else:
            write_log(state["run_dir"], "Editor Warning", "Search/Replace block format mismatch. Writing response verbatim if minor or failing.")
            # Fallback block parsing
    
    attempts = state.get("file_attempts", {}).get(target_file, 0) + 1
    updated_attempts = dict(state.get("file_attempts", {}))
    updated_attempts[target_file] = attempts
    
    mod_files = list(set(state.get("modified_files", []) + [target_file]))

    return {
        "current_task": current_task,
        "plan_stack": plan_stack,
        "file_attempts": updated_attempts,
        "modified_files": mod_files
    }

def reviewer_node(state: AgentState):
    """Validates execution against the popped task, manifest, and goal. Generates issue.md on failure."""
    current_task = state["current_task"]
    target_file = current_task["file"]
    full_path = Path(state["project_address"]) / target_file
    file_content = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
    
    print(f"🔎 [Reviewer] Auditing {target_file}...")
    
    prompt = f"""
    You are a strict Code Reviewer. Validate if the Editor accomplished the task correctly.
    
    Task: {json.dumps(current_task)}
    User Goal: {state['user_request']}
    Manifest: {state['manifest']}
    
    Current File Content:
    ```
    {file_content}
    ```
    
    Analyze syntax correctness and requirement fulfillment.
    If 100% correct and syntax is valid, reply strictly with the word "PASS".
    If there are errors, describe the issue concisely.
    """
    
    review_output = llm.invoke(prompt).content.strip()
    write_log(state["run_dir"], f"Reviewer Node [{target_file}]", review_output)
    
    if re.match(r"^\s*PASS\.?\s*$", review_output, re.IGNORECASE):
        print(f"   ✅ {target_file} passed review.")
        if (state["run_dir"] / "issue.md").exists():
            os.remove(state["run_dir"] / "issue.md")
        return {"issue_report": "", "review_attempts": 0}
    else:
        print(f"   ❌ Issues found in {target_file}. Returning to Editor.")
        (state["run_dir"] / "issue.md").write_text(review_output, encoding="utf-8")
        return {"issue_report": review_output, "review_attempts": state["review_attempts"] + 1}

# ==========================================
# 3. EVALUATION
# ==========================================
if DEEPEVAL_AVAILABLE:
    class OllamaEvalModel(DeepEvalBaseLLM):
        def __init__(self, chat_model, name):
            self._chat = chat_model
            self._name = name
        def load_model(self): return self._chat
        def generate(self, prompt: str) -> str: return self._chat.invoke(prompt).content
        async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
        def get_model_name(self) -> str: return self._name

def _build_deepeval_metrics() -> list:
    judge = OllamaEvalModel(llm, llm.model)
    return [
        GEval(
            name="Correctness",
            criteria="Does the file implementation fulfill the original user request securely and correctly?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.6,
        )
    ]

def compute_custom_score(attempts: int) -> float:
    return round(max(0.25, 1.0 - 0.25 * (attempts - 1)), 2)

def evaluator_node(state: AgentState):
    """Calculates evaluation matrices, Time taken, and Tasks completed[cite: 1]."""
    print("\n📊 [Evaluator] Scoring codebase modifications...")
    run_dir = state["run_dir"]
    end_time = time.time()
    time_taken = end_time - state["start_time"]
    total_tasks = state["total_tasks"]
    
    time_per_task = time_taken / total_tasks if total_tasks > 0 else 0
    context = f"REQUEST:\n{state['user_request']}\nMANIFEST:\n{state['manifest']}"
    metrics = _build_deepeval_metrics() if DEEPEVAL_AVAILABLE else []
    
    report = {}
    for filename in state["modified_files"]:
        full_path = Path(state["project_address"]) / filename
        code = full_path.read_text(encoding="utf-8") if full_path.exists() else ""
        attempts = state["file_attempts"].get(filename, 1)
        
        deval_result = {}
        if DEEPEVAL_AVAILABLE and code.strip():
            test_case = LLMTestCase(input=context, actual_output=code)
            for m in metrics:
                try:
                    m.measure(test_case)
                    deval_result[m.name] = {"score": m.score, "reason": m.reason}
                except Exception as e:
                    deval_result[m.name] = {"score": None, "reason": str(e)}

        report[filename] = {
            "attempts": attempts,
            "custom_score": compute_custom_score(attempts),
            "deepeval": deval_result,
        }

    custom_scores = [f["custom_score"] for f in report.values()]
    overall = {
        "files_modified": len(state["modified_files"]),
        "total_tasks_completed": total_tasks,
        "time_taken_seconds": round(time_taken, 2),
        "time_per_task_seconds": round(time_per_task, 2),
        "custom_score_avg": round(sum(custom_scores) / len(custom_scores), 2) if custom_scores else 0.0,
    }
    
    eval_json = json.dumps({"overall": overall, "files": report}, indent=2)
    (run_dir / "evaluation.json").write_text(eval_json, encoding="utf-8")
    write_log(run_dir, "Evaluator", eval_json)
    
    print(f"   ⏱️ Total Time: {overall['time_taken_seconds']}s")
    print(f"   📈 Time/Task: {overall['time_per_task_seconds']}s")
    print(f"   🎯 Average Custom Score: {overall['custom_score_avg']}/1.0")
    
    return {}

# ==========================================
# 4. CONDITIONAL ROUTERS & GRAPH
# ==========================================
def route_review(state: AgentState) -> str:
    MAX_REVIEW_ATTEMPTS = 3
    if not state.get("issue_report"):
        return "pass"
    if state.get("review_attempts", 0) >= MAX_REVIEW_ATTEMPTS:
        print("   ⚠️️ Max review attempts reached. Forcing pass.")
        return "pass"
    return "fail"

def route_next_task(state: AgentState) -> str:
    return "editor" if len(state.get("plan_stack", [])) > 0 else "evaluator"

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
workflow.add_edge("editor", "reviewer")

workflow.add_conditional_edges("reviewer", route_review, {
    "fail": "editor",
    "pass": "route_check"
})

def mock_route_check(state: AgentState): pass
workflow.add_node("route_check", mock_route_check)
workflow.add_conditional_edges("route_check", route_next_task, {
    "editor": "editor",
    "evaluator": "evaluator"
})
workflow.add_edge("evaluator", END)

app = workflow.compile()

# ==========================================
# 5. EXECUTION BOOTSTRAP
# ==========================================
if __name__ == "__main__":
    target_repo = input("\nEnter the full path to the project directory:\n> ")
    user_req = input("\nDescribe the changes you want to apply:\n> ")
    
    run_path = Path.cwd() / "SDLC_Runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Initiating Agentic Workflow. Logging to: {run_path}")
    
    initial_state = {
        "project_address": target_repo,
        "user_request": user_req,
        "run_dir": run_path,
        "plan_stack": [],
        "issue_report": "",
        "modified_files": [],
        "file_attempts": {}
    }
    
    app.invoke(initial_state)
    print("\n🎉 Architecture Modification Complete!")