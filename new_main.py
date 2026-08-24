# ============================================================
# LOCAL AI SDLC PIPELINE - REACT OFFLINE EDITION
# Gemma 4 + Ollama + LangGraph
# Local Persistence + Safe Code Revision
# ============================================================

import os
import re
import json
import shutil
from datetime import datetime
from typing import TypedDict

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "gemma4:12b"
OLLAMA_URL = "http://localhost:11434"

MAX_REVISIONS = 3

# Persistent Local Directory (Current Working Directory)
BASE_DIR = os.path.join(os.getcwd(), "AI_Agent_Runs")
RUNS_DIR = os.path.join(BASE_DIR, "SDLC_Runs")

os.makedirs(RUNS_DIR, exist_ok=True)


# ============================================================
# 2. STATE
# ============================================================

class WorkflowState(TypedDict, total=False):
    user_input: str
    project_name: str
    requirements: str
    srs: str
    plan: str
    code: str
    critique: str
    pending_tasks: str
    revision_count: int
    status: str
    run_id: str
    run_dir: str
    last_error: str
    code_valid: bool


# ============================================================
# 3. MODEL
# ============================================================

llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_URL,
    temperature=0.1,
    num_ctx=8192,
    num_predict=4096,
)


# ============================================================
# 4. PERSISTENCE HELPERS
# ============================================================

def sanitize_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    return value[:80] or "react_app"

def create_run_directory(user_input: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = sanitize_filename(user_input[:40])
    run_id = f"{timestamp}_{name}"
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def ensure_stage_dir(run_dir, stage):
    path = os.path.join(run_dir, stage)
    os.makedirs(path, exist_ok=True)
    return path

def atomic_write(path: str, content: str):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)

def save_text(run_dir, stage, filename, content):
    path = os.path.join(run_dir, stage, filename)
    atomic_write(path, content or "")
    return path

def save_json(run_dir, stage, filename, data):
    path = os.path.join(run_dir, stage, filename)
    atomic_write(
        path,
        json.dumps(data, indent=2, ensure_ascii=False, default=str)
    )
    return path

def save_checkpoint(state: WorkflowState, node_name: str, updates: dict):
    run_dir = state["run_dir"]
    ensure_stage_dir(run_dir, "checkpoints")
    timestamp = datetime.now().isoformat()
    checkpoint = {
        "node": node_name,
        "timestamp": timestamp,
        "revision_count": updates.get("revision_count", state.get("revision_count", 0)),
        "status": updates.get("status", state.get("status", "")),
        "outputs": updates
    }
    filename = f"{datetime.now().strftime('%H%M%S_%f')}_{node_name}.json"
    save_json(run_dir, "checkpoints", filename, checkpoint)


# ============================================================
# 5. CODE VALIDATION (UPDATED FOR REACT)
# ============================================================

def extract_html(response: str):
    if not response:
        return ""
    response = response.strip()
    match = re.search(r"```html\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if "<html" in candidate.lower(): return candidate
    html_match = re.search(r"<!DOCTYPE html>.*", response, re.DOTALL | re.IGNORECASE)
    if html_match: return html_match.group(0).strip()
    return response.strip()

def validate_html(code: str):
    errors = []
    if not code:
        errors.append("Generated code is EMPTY.")
        return False, errors
    if len(code.strip()) < 100:
        errors.append("Generated code is suspiciously short.")

    lower = code.lower()
    if "<html" not in lower: errors.append("Missing <html> element.")
    if "<body" not in lower: errors.append("Missing <body> element.")
    if "</html>" not in lower: errors.append("HTML appears truncated.")
    if 'id="root"' not in lower: errors.append("Missing <div id='root'> for React mount.")
    if 'type="text/babel"' not in lower: errors.append("Missing <script type='text/babel'> for React logic.")
    
    return len(errors) == 0, errors


# ============================================================
# 6. INITIALIZATION NODE
# ============================================================

def initialize_node(state: WorkflowState):
    user_input = state["user_input"]
    run_id, run_dir = create_run_directory(user_input)
    print("\n" + "=" * 70)
    print("🚀 LOCAL AI SDLC PIPELINE - REACT EDITION")
    print("=" * 70)
    print(f"Model : {MODEL_NAME}")
    print(f"Run   : {run_id}")
    save_text(run_dir, "00_input", "user_request.txt", user_input)
    return {
        "run_id": run_id, "run_dir": run_dir, "revision_count": 0,
        "status": "STARTED", "code": "", "last_error": ""
    }


# ============================================================
# 7. REQUIREMENT ANALYZER
# ============================================================

def input_handler_node(state: WorkflowState):
    print("\n🔍 [Agent 1] Requirement Analyzer")
    prompt = PromptTemplate.from_template(
        "You are a senior Business Analyst.\n"
        "The user wants a React application: {user_input}\n\n"
        "Provide:\n"
        "1. PROJECT_NAME (short, alphanumeric, no spaces)\n"
        "2. REQUIREMENTS (functional, UI, state management)\n\n"
        "Format exactly:\n"
        "PROJECT_NAME: <name>\n"
        "REQUIREMENTS:\n- requirement\n"
    )
    text = llm.invoke(prompt.format(user_input=state["user_input"])).strip()
    match = re.search(r"PROJECT_NAME:\s*([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
    project_name = match.group(1).lower() if match else "react_app"
    updates = {"project_name": project_name, "requirements": text}
    save_text(state["run_dir"], "01_requirements", "requirements.md", text)
    save_checkpoint(state, "analyzer", updates)
    print(f"   Project: {project_name}")
    return updates


# ============================================================
# 8. SRS GENERATOR
# ============================================================

def srs_generator_node(state: WorkflowState):
    print("\n📋 [Agent 2] SRS Generator")
    prompt = PromptTemplate.from_template(
        "Create a Software Requirements Specification for this React App.\n"
        "PROJECT: {project_name}\nREQUIREMENTS: {requirements}\n\n"
        "Include: 1. Purpose 2. Features 3. Component Architecture 4. State Management."
    )
    srs = llm.invoke(prompt.format(
        project_name=state["project_name"], requirements=state["requirements"]
    )).strip()
    updates = {"srs": srs}
    save_text(state["run_dir"], "02_srs", "srs.md", srs)
    save_checkpoint(state, "srs_generator", updates)
    return updates


# ============================================================
# 9. PLANNER
# ============================================================

def planner_node(state: WorkflowState):
    print("\n🗺️ [Agent 3] Planner")
    prompt = PromptTemplate.from_template(
        "You are a senior React architect.\n"
        "Create an implementation plan for a single-file React app based on:\n{srs}\n\n"
        "Cover:\n1. React Components (functional)\n2. State hooks (useState, useEffect)\n"
        "3. Styling (inline CSS or tailwind via CDN if needed)\n4. Error Handling."
    )
    plan = llm.invoke(prompt.format(srs=state["srs"])).strip()
    updates = {"plan": plan}
    save_text(state["run_dir"], "03_plan", "plan.md", plan)
    save_checkpoint(state, "planner", updates)
    return updates


# ============================================================
# 10. CODER
# ============================================================

def coder_node(state: WorkflowState):
    revision = state.get("revision_count", 0)
    previous_code = state.get("code", "")
    print(f"\n💻 [Agent 4] Coder (Pass {revision + 1})")

    strict_template = """
OUTPUT REQUIREMENTS: Return ONLY ONE complete HTML document.

Required structure EXACTLY like this:
<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
  <style>
    /* ALL CSS HERE */
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel">
    const { useState, useEffect } = React;
    
    // ALL REACT COMPONENTS AND APP LOGIC HERE
    
    const root = ReactDOM.createRoot(document.getElementById('root'));
    root.render(<App />);
  </script>
</body>
</html>

Rules:
1. Do NOT use markdown outside the HTML.
2. Ensure ALL tags are closed.
3. NEVER intentionally truncate the response.
"""

    if revision == 0:
        prompt = PromptTemplate.from_template(
            "Build a functional single-page React web application.\n"
            "PLAN: {plan}\n{strict_template}"
        )
        formatted = prompt.format(plan=state["plan"], strict_template=strict_template)
    else:
        prompt = PromptTemplate.from_template(
            "Fix the existing React application.\n"
            "PLAN: {plan}\nCURRENT CODE:\n{code}\nDEFECTS:\n{critique}\n"
            "Return the COMPLETE application again.\n{strict_template}"
        )
        formatted = prompt.format(
            plan=state["plan"], code=previous_code, 
            critique=state.get("critique", ""), strict_template=strict_template
        )

    response = llm.invoke(formatted)
    generated_code = extract_html(response if response else "")
    valid, errors = validate_html(generated_code)

    attempt = revision + 1
    attempt_dir = os.path.join(state["run_dir"], "04_code", f"attempt_{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)

    save_text(state["run_dir"], "04_code", f"raw_attempt_{attempt}.txt", response if response else "")
    save_text(state["run_dir"], f"04_code/attempt_{attempt}", "candidate.html", generated_code)
    save_json(
        state["run_dir"], f"04_code/attempt_{attempt}", "validation.json",
        {"valid": valid, "errors": errors, "length": len(generated_code), "timestamp": datetime.now().isoformat()}
    )

    if valid:
        save_text(state["run_dir"], "04_code", "latest_valid.html", generated_code)
        updates = {"code": generated_code, "revision_count": attempt, "code_valid": True, "last_error": ""}
        print("   ✅ Valid React code generated.")
    else:
        print("   ⚠️ Generated code FAILED validation.")
        for error in errors: print(f"      - {error}")
        if previous_code:
            print("   🛡️ Keeping previous valid code.")
            updates = {"code": previous_code, "revision_count": attempt, "code_valid": False, "last_error": "\n".join(errors)}
        else:
            updates = {"code": "", "revision_count": attempt, "code_valid": False, "last_error": "\n".join(errors)}

    save_checkpoint(state, "coder", updates)
    return updates


# ============================================================
# 11. REVIEWER
# ============================================================

def reviewer_node(state: WorkflowState):
    revision = state.get("revision_count", 1)
    print(f"\n🔎 [Agent 5] QA Auditor (Pass {revision})")

    code = state.get("code", "")
    structural_valid, structural_errors = validate_html(code)

    if not structural_valid:
        critique = "PROGRAMMATIC QA FAILURE:\n" + "\n".join(f"- {e}" for e in structural_errors)
        updates = {"status": "NEEDS_REVISION", "critique": critique, "pending_tasks": "\n".join(structural_errors)}
        save_text(state["run_dir"], "05_qa", f"qa_pass_{revision}.md", critique)
        save_checkpoint(state, "reviewer", updates)
        return updates

    prompt = PromptTemplate.from_template(
        "You are a strict QA engineer. Evaluate this React application.\n"
        "REQUIREMENTS: {requirements}\nCODE: {code}\n"
        "Check: 1. React errors 2. UI functionality 3. Missing features.\n"
        "Respond exactly:\nSTATUS: APPROVED or NEEDS_REVISION\nPENDING_TASKS:\n- task\nCRITIQUE:\n- issue"
    )
    review = llm.invoke(prompt.format(requirements=state["requirements"], code=code)).strip()
    
    approved = "STATUS: APPROVED" in review.upper()
    if approved:
        status, pending = "APPROVED", "None"
    else:
        status = "NEEDS_REVISION"
        pending_match = re.search(r"PENDING_TASKS:\s*(.*?)(?=CRITIQUE:|$)", review, re.DOTALL | re.IGNORECASE)
        pending = pending_match.group(1).strip() if pending_match else "Review identified issues."

    updates = {"status": status, "critique": review, "pending_tasks": pending}
    save_text(state["run_dir"], "05_qa", f"qa_pass_{revision}.md", review)
    save_checkpoint(state, "reviewer", updates)
    print(f"   ↳ QA Status: {status}")
    return updates


# ============================================================
# 12. REVISION ROUTER & SAVER
# ============================================================

def check_revision_condition(state: WorkflowState):
    if state.get("status", "") == "NEEDS_REVISION" and state.get("revision_count", 0) < MAX_REVISIONS:
        return "reiterate"
    return "finalize"

def file_writer_node(state: WorkflowState):
    print("\n💾 [Agent 6] Final Artifact Writer")
    run_dir, final_dir = state["run_dir"], os.path.join(state["run_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    final_code = state.get("code", "")
    valid, _ = validate_html(final_code)

    if not valid:
        fallback = os.path.join(run_dir, "04_code", "latest_valid.html")
        if os.path.exists(fallback):
            print("   🛡️ Restoring latest valid code.")
            shutil.copy2(fallback, os.path.join(final_dir, "index.html"))
        else:
            raise RuntimeError("Pipeline finished without a valid React artifact.")
    else:
        save_text(run_dir, "final", "index.html", final_code)

    save_text(run_dir, "final", "requirements.md", state.get("requirements", ""))
    save_text(run_dir, "final", "srs.md", state.get("srs", ""))
    save_text(run_dir, "final", "plan.md", state.get("plan", ""))
    save_text(run_dir, "final", "qa_review.md", state.get("critique", ""))
    print("\n🎉 BUILD COMPLETE")
    print(f"📁 Run directory:\n{run_dir}")
    return {}


# ============================================================
# 13. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(WorkflowState)
workflow.add_node("initialize", initialize_node)
workflow.add_node("analyzer", input_handler_node)
workflow.add_node("srs_generator", srs_generator_node)
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("saver", file_writer_node)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "analyzer")
workflow.add_edge("analyzer", "srs_generator")
workflow.add_edge("srs_generator", "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_conditional_edges("reviewer", check_revision_condition, {"reiterate": "coder", "finalize": "saver"})
workflow.add_edge("saver", END)
app = workflow.compile()

# ============================================================
# 14. RUN
# ============================================================
if __name__ == "__main__":
    user_prompt = input("\nEnter React application requirement: ")
    app.invoke({"user_input": user_prompt})
    print("\nPipeline finished.")