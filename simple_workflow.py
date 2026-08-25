import re
from pathlib import Path
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# Initialize LLM for local execution
llm = ChatOllama(
    model="gemma4:e2b",
    num_predict=2048,
    num_ctx=4096,  # Adjust based on your local GPU VRAM
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
# 2. STATE DEFINITION
# ==========================================
class AgentState(TypedDict):
    user_input: str
    requirements: str
    run_dir: Path
    file_system: Annotated[dict, operator.ior]
    pending_files: list[str]
    iterations: int
    feedback: str
    review_attempts: int


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

    req_file = state["run_dir"] / "requirements.txt"
    req_file.write_text(requirements, encoding="utf-8")

    return {"requirements": requirements}


def architect_node(state: AgentState):
    print("\n📐 [Architect] Structuring project files...")
    initial_files = {
        "index.html": "<!DOCTYPE html>\n<html lang='en'>\n<head>\n  <meta charset='UTF-8'>\n  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n  <title>App</title>\n  <link rel='stylesheet' href='style.css'>\n</head>\n<body>\n  <div id='app'>\n    <!-- UI Layout -->\n  </div>\n  <script src='script.js'></script>\n</body>\n</html>",
        "style.css": "/* App Styles */\nbody { font-family: sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; }",
        "script.js": "// App Operations\nconsole.log('App initialized');"
    }
    write_log(state["run_dir"], "Architect Node", f"Scaffolded files: {list(initial_files.keys())}")
    return {
        "file_system": initial_files,
        "pending_files": ["index.html", "style.css", "script.js"],
        "iterations": 0
    }


def editor_node(state: AgentState):
    current_file = state["pending_files"][0]

    print(f"\n💻 [Editor] Drafting {current_file} (Streaming)...")

    # NEW: Build a string containing the current state of all files
    project_context = ""
    for filename, content in state["file_system"].items():
        project_context += f"\n--- CURRENT {filename.upper()} ---\n{content}\n"

    prompt = f"""
    You are an expert web developer. Write the code for {current_file}.

    REQUIREMENTS:
    {state['requirements']}

    CURRENT PROJECT STATE (Use this to match IDs, classes, and structure):
    {project_context}

    Provide ONLY the complete, updated code for {current_file}. Do not include markdown formatting or explanations.
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

    log_message = f"Length: {len(raw_output)} chars\n\n{raw_output}" if raw_output else "[ERROR] LLM returned 0 characters."
    write_log(state["run_dir"], f"Editor Node [{current_file}] - RAW OUTPUT", log_message)

    fence_match = re.search(r"```(?:[a-zA-Z]+)?\n?(.*?)\n?```", raw_output, re.DOTALL)
    clean_code = fence_match.group(1).strip() if fence_match else raw_output.strip()

    updated_fs = state["file_system"].copy()

    if len(clean_code) > 10:
        updated_fs[current_file] = clean_code
        print(f"   ✓ {current_file} updated ({len(clean_code)} chars)")
        write_log(state["run_dir"], f"Editor Node [{current_file}] - ACTION", "Successfully updated file_system.")
    else:
        print(f"   ⚠️ Model returned empty code. Retaining template for {current_file}.")
        write_log(state["run_dir"], f"Editor Node [{current_file}] - ACTION", "Fallback triggered. Retained base code.")

    return {
        "file_system": updated_fs,
        "iterations": state["iterations"] + 1
    }


def reviewer_node(state: AgentState):
    current_file = state["pending_files"][0]
    code_to_review = state["file_system"][current_file]

    print(f"\n🔎 [Reviewer] Checking {current_file}...")

    prompt = f"""
    Review this {current_file} for a billing app.
    REQUIREMENTS: {state['requirements']}
    CODE: {code_to_review}

    Does this code fully meet the requirements and use robust logic? 
    If yes, reply ONLY with "PASS". 
    If no, list the specific bugs or missing features.
    """

    review_output = llm.invoke(prompt).content.strip()

    if "PASS" in review_output.upper() or state["review_attempts"] >= 2:
        print("   ✅ Code approved (or max attempts reached).")
        return {
            "pending_files": state["pending_files"][1:],
            "feedback": "",
            "review_attempts": 0
        }
    else:
        print(f"   ❌ Issues found. Sending back to Editor.")
        return {
            "feedback": review_output,
            "review_attempts": state["review_attempts"] + 1
        }


def saver_node(state: AgentState):
    print("\n💾 [Saver] Writing generated files to disk...")
    save_log = []
    for filename, content in state["file_system"].items():
        file_path = state["run_dir"] / filename
        file_path.write_text(content, encoding="utf-8")
        msg = f"Saved: {file_path} ({len(content)} chars)"
        print(f"   💾 {msg}")
        save_log.append(msg)
    write_log(state["run_dir"], "Saver Node", "\n".join(save_log))
    return {}


# ==========================================
# 4. CONDITIONAL ROUTER & GRAPH
# ==========================================
def should_continue(state: AgentState):
    if len(state["pending_files"]) > 0:
        return "continue"
    return "save"


workflow = StateGraph(AgentState)
workflow.add_node("requirements", requirement_node)
workflow.add_node("architect", architect_node)
workflow.add_node("editor", editor_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("saver", saver_node)

workflow.set_entry_point("requirements")
workflow.add_edge("requirements", "architect")
workflow.add_edge("architect", "editor")
workflow.add_edge("editor", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue, {"continue": "editor", "save": "saver"})
workflow.add_edge("saver", END)
app = workflow.compile()

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    user_input = input("\nWhat application do you want to build?\n> ")

    # Switched to local directory path mapping
    base_dir = Path.cwd() / "SDLC_Runs"
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    write_log(run_dir, "SYSTEM", f"Starting build for: '{user_input}'\nModel: gemma4:12b")

    print(f"\n🚀 Starting build in: {run_dir}")
    app.invoke({"user_input": user_input, "run_dir": run_dir})
    print(f"\n🎉 Build Complete! Check execution_log.txt in {run_dir}")