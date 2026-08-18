import os
import re
from typing import TypedDict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------
class WorkflowState(TypedDict):
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

# ---------------------------------------------------------
# 2. Local Model Configuration
# ---------------------------------------------------------
llm = Ollama(
    model="gemma4:e2b", 
    base_url="http://localhost:11434", 
    temperature=0.1,
    num_predict=4096 
)
MAX_REVISIONS = 3

# ---------------------------------------------------------
# 3. Agent Nodes
# ---------------------------------------------------------

def input_handler_node(state: WorkflowState):
    print("\n🔍 [Agent 1] Requirement Analyzer: Structuring core features...")
    prompt = PromptTemplate.from_template(
        "You are a Lead Business Analyst. The user wants to build: '{user_input}'.\n\n"
        "Provide:\n"
        "1. A single-word alphanumeric PROJECT_NAME.\n"
        "2. A structured list of MUST-HAVE visual and interactive features.\n\n"
        "Format:\n"
        "PROJECT_NAME: <single_word_name>\n"
        "REQUIREMENTS:\n<bulleted features>"
    )
    response = llm.invoke(prompt.format(user_input=state["user_input"]))
    
    match = re.search(r"PROJECT_NAME:\s*([a-zA-Z0-9_-]+)", response)
    project_name = match.group(1).strip().lower() if match else "web_app"
    
    return {"project_name": project_name, "requirements": response, "revision_count": 0}


def srs_generator_node(state: WorkflowState):
    print("📋 [Agent 2] SRS Generator: Drafting technical specifications...")
    prompt = PromptTemplate.from_template(
        "Write a concise Software Requirements Specification (SRS) based on:\n{requirements}\n"
    )
    return {"srs": llm.invoke(prompt.format(requirements=state["requirements"]))}


def planner_node(state: WorkflowState):
    print("🗺️ [Agent 3] Planner: Creating implementation checklist...")
    prompt = PromptTemplate.from_template(
        "Based on this SRS:\n{srs}\n\n"
        "Create a strict technical checklist covering HTML layout, CSS styling, and JavaScript DOM logic."
    )
    return {"plan": llm.invoke(prompt.format(srs=state["srs"]))}


def coder_node(state: WorkflowState):
    revision = state.get("revision_count", 0)
    
    # Enhanced strict template to prevent truncation and limit CSS bloat
    strict_template = (
        "You MUST output ONE single file using this EXACT structure. Do not deviate:\n"
        "```html\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "  <style>\n"
        "    /* KEEP CSS EXTREMELY BRIEF AND MINIMAL. Do not waste tokens on heavy styling. */\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <!-- ALL HTML GOES HERE -->\n"
        "\n"
        "  <script>\n"
        "    // ALL JAVASCRIPT LOGIC MUST GO HERE.\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
        "```\n"
        "CRITICAL RULES:\n"
        "1. NEVER use <link href=> or <script src=>.\n"
        "2. KEEP CSS MINIMAL. Your primary goal is functional JavaScript.\n"
        "3. DO NOT TRUNCATE THE CODE. You must complete the <script> tags before stopping."
    )

    
    if revision == 0:
        print("\n💻 [Agent 4] Coder: Generating initial complete application [Initial Pass]...")
        prompt = PromptTemplate.from_template(
            "Write a production-ready single-page application.\n\n"
            "Plan:\n{plan}\n\n" + strict_template
        )
        code = llm.invoke(prompt.format(plan=state["plan"]))
    else:
        print(f"\n🔄 [Agent 4] Coder: Applying QA revisions [Loop {revision}/{MAX_REVISIONS}]...")
        prompt = PromptTemplate.from_template(
            "Fix the application based on QA feedback.\n\n"
            "CURRENT CODE:\n```html\n{code}\n```\n\n"
            "DEFECTS TO FIX:\n{critique}\n\n" + strict_template
        )
        code = llm.invoke(prompt.format(code=state["code"], critique=state["critique"]))

    match = re.search(r"```html(.*?)```", code, re.DOTALL | re.IGNORECASE)
    clean_code = match.group(1).strip() if match else code.strip()
    if clean_code.startswith("```"): clean_code = clean_code.replace("```html", "").replace("```", "")
    
    return {"code": clean_code, "revision_count": revision + 1}


def reviewer_node(state: WorkflowState):
    revision = state.get("revision_count", 1)
    print(f"🔎 [Agent 5] QA Auditor: Inspecting code completeness [Evaluating Loop {revision}/{MAX_REVISIONS}]...")
    
    # AI Review Phase
    prompt = PromptTemplate.from_template(
        "You are a strict code auditor. Evaluate this code:\n\n```html\n{code}\n```\n\n"
        "RULES TO ENFORCE:\n"
        "1. MUST contain <style> tags with CSS.\n"
        "2. MUST contain <script> tags with JavaScript logic.\n"
        "3. MUST NOT contain <link rel='stylesheet'> or <script src=>.\n\n"
        "NEVER suggest moving CSS or JS to external files.\n\n"
        "Respond EXACTLY in this format:\n"
        "STATUS: <APPROVED or NEEDS_REVISION>\n"
        "PENDING_TASKS:\n<List exact failures>\n"
        "CRITIQUE:\n<Instructions to fix>"
    )
    
    review = llm.invoke(prompt.format(code=state["code"]))
    
    # ---------------------------------------------------------
    # PROGRAMMATIC SYSTEM OVERRIDE (Safeguard against AI hallucinations)
    # ---------------------------------------------------------
    raw_code = state["code"].lower()
    
    missing_js = "<script>" not in raw_code
    has_external_links = "<link rel=" in raw_code or "<script src=" in raw_code
    
    # If Python detects a structural flaw, override the AI's "APPROVED" status
    if missing_js or has_external_links:
        status = "NEEDS_REVISION"
        system_critique = "\n\n--- SYSTEM OVERRIDE ERRORS ---\n"
        
        if missing_js:
            system_critique += "FATAL ERROR: The <script> tags are completely missing or truncated. You MUST embed the full JavaScript logic inside <script> tags.\n"
        if has_external_links:
            system_critique += "FATAL ERROR: External files detected. Remove all <link> and <script src=> tags. Embed all CSS in <style> and JS in <script>.\n"
            
        review += system_critique
        print("   ⚠️ [System Override Triggered: Hard failures detected in code structure]")
    else:
        # If structure is fundamentally sound, trust the AI's review status
        status = "APPROVED" if "STATUS: APPROVED" in review.upper() else "NEEDS_REVISION"
    
    # Extract pending tasks
    pending_match = re.search(r"PENDING_TASKS:\s*(.*?)(?=CRITIQUE:|$)", review, re.DOTALL | re.IGNORECASE)
    pending = pending_match.group(1).strip() if pending_match else "None"
    
    print(f"   ↳ Final Audit Status: {status} | Pending Items: {'Yes' if status == 'NEEDS_REVISION' else 'None'}")
    
    return {"status": status, "critique": review, "pending_tasks": pending}

def file_writer_node(state: WorkflowState):
    print("\n💾 [Agent 6] File Writer: Saving final validated artifacts...")
    project_dir = os.path.join("output", state["project_name"])
    os.makedirs(project_dir, exist_ok=True)
    
    files = {
        "requirements.md": state.get("requirements", ""),
        "srs.md": state.get("srs", ""),
        "plan.md": state.get("plan", ""),
        "qa_review_log.md": state.get("critique", ""),
        "index.html": state.get("code", "")
    }
    
    for filename, content in files.items():
        with open(os.path.join(project_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
            
    print(f"🎉 Build Complete! Output saved to: ./{project_dir}/")
    return {}

# ---------------------------------------------------------
# 4. Conditional Edge & Compilation
# ---------------------------------------------------------
def check_revision_condition(state: WorkflowState):
    if state["status"] == "NEEDS_REVISION" and state["revision_count"] <= MAX_REVISIONS:
        return "reiterate"
    return "finalize"

workflow = StateGraph(WorkflowState)
workflow.add_node("analyzer", input_handler_node)
workflow.add_node("srs_generator", srs_generator_node)
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("saver", file_writer_node)

workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "srs_generator")
workflow.add_edge("srs_generator", "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "reviewer")

workflow.add_conditional_edges("reviewer", check_revision_condition, {"reiterate": "coder", "finalize": "saver"})
workflow.add_edge("saver", END)

app = workflow.compile()

if __name__ == "__main__":
    user_prompt = input("Enter application requirement: ")
    app.invoke({"user_input": user_prompt})