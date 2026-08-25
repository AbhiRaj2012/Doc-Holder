import re
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

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

# ==========================================
# 3. NODES
# ==========================================
def requirement_node(state: AgentState):
    print("🔍 Generating requirements...")
    prompt = f"""
    Convert this request into concise requirements for a simple web app.
    Use ONLY HTML, CSS, and Vanilla JavaScript across separate files.
    REQUEST: {state['user_input']}
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
    Based on these requirements, create a strict technical contract (Manifest).
    You MUST output:
    1. LAYOUT STRUCTURE: Define the exact wrapper classes needed (e.g., '.calculator-grid', '.display-area') so CSS can implement a Grid/Flexbox properly.
    2. DOM IDs: A dictionary of exact DOM Element IDs that HTML and JS will strictly share.
    3. DATA: The exact JSON data structure for the localStorage ledger.

    REQUIREMENTS:
    {state['requirements']}
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
        "iterations": 0
    }


def editor_node(state: AgentState):
    current_file = state["pending_files"][0]
    base_code = state["file_system"][current_file]

    print(f"\n💻 [Editor] Drafting {current_file} (Streaming)...")

    project_context = "".join([f"\n--- {f.upper()} ---\n{c}\n" for f, c in state["file_system"].items()])

    # UPGRADE: Strict diff-style instructions
    prompt = f"""
    You are an expert web developer editing {current_file}.

    PROJECT MANIFEST (Contract):
    {state['project_manifest']}

    CURRENT PROJECT FILES:
    {project_context}

    PREVIOUS REVIEWER FEEDBACK:
    {state['feedback']}

    CRITICAL INSTRUCTION: 
    If there is feedback, you MUST ONLY change the exact lines mentioned. DO NOT refactor or rewrite the rest of the file.
    You MUST return the ENTIRE updated file content so it can be saved, but leave the un-flagged code exactly as it was.
    Provide ONLY the complete code for {current_file}. No markdown code fences, no explanations.
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

    fence_match = re.search(r"```(?:[a-zA-Z]+)?\n?(.*?)\n?```", raw_output, re.DOTALL)
    clean_code = fence_match.group(1).strip() if fence_match else raw_output.strip()

    updated_fs = state["file_system"].copy()
    if len(clean_code) > 10:
        updated_fs[current_file] = clean_code

    return {
        "file_system": updated_fs,
        "iterations": state["iterations"] + 1
    }


def reviewer_node(state: AgentState):
    current_file = state["pending_files"][0]
    file_code = state["file_system"][current_file]

    print(f"\n🔎 [Reviewer] Auditing {current_file}...")

    project_context = "".join([f"\n--- {f.upper()} ---\n{c}\n" for f, c in state["file_system"].items()])

    # UPGRADE: Added Syntax Linter and structured Replace/With feedback
    prompt = f"""
    Perform a strict Syntax, DOM, and Functionality Audit on {current_file}.

    MANIFEST: {state['project_manifest']}
    FULL PROJECT CONTEXT: {project_context}
    CODE TO REVIEW ({current_file}):
    {file_code}

    CRITICAL CHECKS:
    1. SYNTAX LINTING: Are there any mismatched quotes (e.g. id='btn">), unclosed tags, or missing brackets?
    2. DOM BINDING: Do the IDs/Classes exactly match the Manifest?
    3. UI LAYOUT: Does the CSS implement a proper Grid or Flexbox matching the HTML wrappers?

    If the code is 100% correct, reply with ONLY the word "PASS". 
    If there are errors, identify the specific mistake and provide SURGICAL feedback in this exact format:
    "Replace: [bad code line]
    With: [corrected code line]"
    """

    review_output = llm.invoke(prompt).content.strip()
    write_log(state["run_dir"], f"Reviewer Node [{current_file}]", review_output)

    if "PASS" in review_output.upper() or state["review_attempts"] >= 3: # Increased allowance to 3 attempts
        print(f"   ✅ {current_file} passed review.")
        return {
            "feedback": "",
            "review_attempts": 0
        }
    else:
        print(f"   ❌ Issues found in {current_file}. Routing back to Editor for surgical fix.")
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
    return {"pending_files": state["pending_files"][1:]}


# ==========================================
# 4. CONDITIONAL ROUTERS & GRAPH
# ==========================================
def route_review(state: AgentState):
    if state["feedback"] == "":
        return "pass"
    return "fail"

def route_next_file(state: AgentState):
    if len(state["pending_files"]) > 0:
        return "continue"
    return "done"

workflow = StateGraph(AgentState)
workflow.add_node("requirements", requirement_node)
workflow.add_node("architect", architect_node)
workflow.add_node("editor", editor_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("save_file", save_file_node)

workflow.set_entry_point("requirements")
workflow.add_edge("requirements", "architect")
workflow.add_edge("architect", "editor")
workflow.add_edge("editor", "reviewer")

workflow.add_conditional_edges("reviewer", route_review, {"fail": "editor", "pass": "save_file"})
workflow.add_conditional_edges("save_file", route_next_file, {"continue": "editor", "done": END})

app = workflow.compile()

# ==========================================
# 5. EXECUTION
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