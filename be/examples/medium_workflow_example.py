from typing import TypedDict, List, Annotated
import operator
import time # For small delays to make logs distinct
from datetime import datetime # For timestamps in custom logs

# Adjust the import path based on your project structure
# This assumes spectra.py is in a directory 'spectra' adjacent to 'examples'
# and your PYTHONPATH is set up correctly, or you run this from the 'be' directory.
from spectra.spectra import Spectra 
from langgraph.graph import StateGraph, END

# 1. Define the state for the workflow
class MediumWorkflowState(TypedDict):
    input_text: str
    processing_steps: List[str]
    analysis_result_alpha: str
    analysis_result_beta: str
    enrichment_data: dict
    synthesis_output: str
    final_decision: str
    error_log: str

# --- Spectra Logger Container ---
# Node functions will close over this mutable dictionary, allowing late binding of the instance.
logger_container = {'instance': None}

# 2. Define node functions
# These functions will access the logger via logger_container['instance']

def ingestion_node(state: MediumWorkflowState) -> MediumWorkflowState:
    print("--- Node: Ingestion ---")
    current_step = "Ingested input text."
    state['processing_steps'].append(current_step)
    print(f"  Input: {state['input_text']}")
    print(f"  Action: {current_step}")
    
    if logger_container['instance']:
        logger_container['instance']._log_to_jsonl({
            "event": "ingestion_custom_log",
            "timestamp": datetime.now().isoformat(),
            "node_name": "ingestion",
            "detail": "Input text successfully ingested and initial state populated.",
            "input_length": len(state['input_text'])
        })
        time.sleep(0.1)
    return {
        "processing_steps": state['processing_steps']
    }

def analysis_alpha_node(state: MediumWorkflowState) -> MediumWorkflowState:
    print("--- Node: Analysis Alpha ---")
    current_step = "Performed alpha analysis on data."
    result = f"Alpha analysis of '{state['input_text'][:20]}...' completed."
    state['processing_steps'].append(current_step)
    print(f"  Action: {current_step}")
    print(f"  Result: {result}")

    if logger_container['instance']:
        for i in range(4): # Generate more logs
            logger_container['instance']._log_to_jsonl({
                "event": "alpha_analysis_sub_step",
                "timestamp": datetime.now().isoformat(),
                "node_name": "analysis_alpha",
                "step_number": i + 1,
                "detail": f"Analyzing segment {i+1} of alpha phase.",
                "current_result_so_far": result
            })
            time.sleep(0.1) 

    return {
        "processing_steps": state['processing_steps'],
        "analysis_result_alpha": result
    }

def analysis_beta_node(state: MediumWorkflowState) -> MediumWorkflowState:
    print("--- Node: Analysis Beta ---")
    current_step = "Performed beta analysis on data."
    result = f"Beta analysis based on '{state['analysis_result_alpha']}' completed."
    state['processing_steps'].append(current_step)
    print(f"  Action: {current_step}")
    print(f"  Result: {result}")
    if logger_container['instance']:
        logger_container['instance']._log_to_jsonl({
            "event": "beta_analysis_complete",
            "timestamp": datetime.now().isoformat(),
            "node_name": "analysis_beta",
            "detail": "Beta analysis phase concluded.",
            "based_on_alpha_result": state['analysis_result_alpha']
        })
        time.sleep(0.1)
    return {
        "processing_steps": state['processing_steps'],
        "analysis_result_beta": result
    }

def fetch_external_details_tool(query: str, previous_analysis: str) -> dict:
    if logger_container['instance']:
        logger_container['instance']._log_to_jsonl({
            "event": "tool_call_initiated", 
            "timestamp": datetime.now().isoformat(), 
            "tool_name": "fetch_external_details_tool", 
            "query": query,
            "context_summary": previous_analysis[:30]
        })
        time.sleep(0.1)
    print(f"    🛠️ Tool Call: fetch_external_details_tool")
    print(f"      Tool Input Query: {query}")
    print(f"      Tool Context: Based on '{previous_analysis[:30]}...'")
    if "complex data" in query.lower():
        details = {"source": "SimulatedExternalAPI", "confidence": 0.92, "details": "Detailed report on complex data patterns found.", "items_found": 5, "status": "OK"}
    else:
        details = {"source": "SimulatedExternalAPI", "confidence": 0.75, "details": "Standard enrichment applied.", "items_found": 2, "status": "OK"}
    if logger_container['instance']:
        logger_container['instance']._log_to_jsonl({
            "event": "tool_call_completed", 
            "timestamp": datetime.now().isoformat(), 
            "tool_name": "fetch_external_details_tool", 
            "result_summary": details['details']
        })
        time.sleep(0.1)
    return details

def data_enrichment_node(state: MediumWorkflowState) -> MediumWorkflowState:
    print("--- Node: Data Enrichment (with Tool Call) ---")
    current_step = "Preparing for data enrichment tool call."
    state['processing_steps'].append(current_step)
    print(f"  Action: {current_step}")

    tool_query = state['input_text']
    tool_context = state['analysis_result_beta']
    
    all_enriched_info = []
    # Ensure logger is available for logging loop, otherwise, this node won't log enrichment steps
    if logger_container['instance']:
        for i in range(2): # Simulate multiple enrichment calls or aspects
            logger_container['instance']._log_to_jsonl({
                "event": "enrichment_iteration_start",
                "timestamp": datetime.now().isoformat(),
                "node_name": "enrichment",
                "iteration": i + 1,
                "query_focus": f"Focus on aspect {i+1} of query: {tool_query[:30]}..."
            })
            time.sleep(0.1)
            enriched_info_part = fetch_external_details_tool(query=f"{tool_query} (aspect {i+1})", previous_analysis=tool_context)
            all_enriched_info.append(enriched_info_part)
            current_step_after_tool = f"Data enrichment part {i+1} completed. Source: {enriched_info_part.get('source')}, Items: {enriched_info_part.get('items_found')}."
            state['processing_steps'].append(current_step_after_tool)
            print(f"  Tool Output (part {i+1}): {enriched_info_part}")
            print(f"  Action: {current_step_after_tool}")
            time.sleep(0.1)
    else:
        # Fallback: If logger somehow wasn't initialized, perform minimal operation without detailed logging
        # This helps identify if the logger_container pattern failed, though it shouldn't.
        print("*** Warning: Spectra logger not available in data_enrichment_node. Performing minimal enrichment. ***")
        enriched_info_part = fetch_external_details_tool(query=f"{tool_query} (aspect 1)", previous_analysis=tool_context)
        all_enriched_info.append(enriched_info_part)
        current_step_after_tool = f"Data enrichment part 1 (minimal due to no logger) completed. Source: {enriched_info_part.get('source')}, Items: {enriched_info_part.get('items_found')}."
        state['processing_steps'].append(current_step_after_tool)

    final_enrichment_summary = all_enriched_info[-1] if all_enriched_info else {}
    return {
        "processing_steps": state['processing_steps'],
        "enrichment_data": final_enrichment_summary 
    }

def synthesis_node(state: MediumWorkflowState) -> MediumWorkflowState:
    print("--- Node: Synthesis ---")
    current_step = "Synthesized all findings."
    enrichment_details = state.get('enrichment_data', {}).get('details', "No enrichment details")
    synthesis = f"Synthesized: Alpha ({state['analysis_result_alpha']}), Beta ({state['analysis_result_beta']}), Enrichment ({enrichment_details})"
    state['processing_steps'].append(current_step)
    print(f"  Action: {current_step}")
    print(f"  Synthesis: {synthesis[:100]}...")

    if logger_container['instance']:
        for i in range(3): # Generate more logs
            logger_container['instance']._log_to_jsonl({
                "event": "synthesis_sub_step",
                "timestamp": datetime.now().isoformat(),
                "node_name": "synthesis",
                "step_number": i + 1,
                "detail": f"Synthesizing aspect {i+1}. Current synthesis: {synthesis[:50]}..."
            })
            time.sleep(0.1)

    return {
        "processing_steps": state['processing_steps'],
        "synthesis_output": synthesis
    }

def decision_node(state: MediumWorkflowState) -> MediumWorkflowState:
    print("--- Node: Decision ---")
    current_step = "Made final decision based on synthesis."
    decision = "Proceed with action plan Alpha" if "Alpha" in state['synthesis_output'] else "Review required for action plan Beta"
    state['processing_steps'].append(current_step)
    print(f"  Action: {current_step}")
    print(f"  Decision: {decision}")
    if logger_container['instance']:
        logger_container['instance']._log_to_jsonl({
            "event": "final_decision_logged",
            "timestamp": datetime.now().isoformat(),
            "node_name": "decision",
            "decision_made": decision,
            "basis": state['synthesis_output']
        })
        time.sleep(0.1)
    return {
        "processing_steps": state['processing_steps'],
        "final_decision": decision
    }

# 3. Construct the StateGraph
workflow_builder = StateGraph(MediumWorkflowState)

# Add nodes using the functions defined above
workflow_builder.add_node("ingestion", ingestion_node)
workflow_builder.add_node("analysis_alpha", analysis_alpha_node)
workflow_builder.add_node("analysis_beta", analysis_beta_node)
workflow_builder.add_node("enrichment", data_enrichment_node)
workflow_builder.add_node("synthesis", synthesis_node)
workflow_builder.add_node("decision", decision_node)

# Set the entrypoint and edges
workflow_builder.set_entry_point("ingestion")
workflow_builder.add_edge("ingestion", "analysis_alpha")
workflow_builder.add_edge("analysis_alpha", "analysis_beta")
workflow_builder.add_edge("analysis_beta", "enrichment")
workflow_builder.add_edge("enrichment", "synthesis")
workflow_builder.add_edge("synthesis", "decision")
workflow_builder.add_edge("decision", END)

# 4. Initialize Spectra logger instance for real
print("🚀 Initializing Spectra for Medium Workflow...")
logger_container['instance'] = Spectra( # Assign to the 'instance' key in the container
    workflow=workflow_builder, 
    output_dir="logs/medium_workflow",
    live_logging=True, 
    capture_thinking=True, 
    auto_process=True 
)
print("✅ Spectra initialized.")


# 5. Define an initial state and invoke the workflow
initial_state: MediumWorkflowState = {
    "input_text": "This is a complex piece of data requiring thorough multistep analysis and enrichment before a final decision.",
    "processing_steps": [],
    "analysis_result_alpha": "",
    "analysis_result_beta": "",
    "enrichment_data": {},
    "synthesis_output": "",
    "final_decision": "",
    "error_log": ""
}

print(f"🏁 Running Medium Workflow with Spectra via .run()...\nInitial State: {initial_state['input_text']}")

try:
    # Access the run method via the container
    final_state = logger_container['instance'].run(initial_state)
    print("🎉 Medium Workflow execution completed with Spectra.")
    print("--------------------------------------------------")
    print("📊 Final State:")
    for key, value in final_state.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    print("--------------------------------------------------")
except Exception as e:
    print(f"💥 An error occurred during workflow execution: {e}")

print("\nℹ️ Check the 'logs/medium_workflow' directory for detailed execution logs.")
print("ℹ️ Check your Spectra dashboard at http://localhost:3000 for visualization.") 