from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from spectra import Spectra # Assuming spectra is in PYTHONPATH or installed
from dotenv import load_dotenv

load_dotenv()

# Define state
class BasicAgentState(TypedDict):
    message: Annotated[str, "The input message"]
    response: Annotated[str, "The agent's response"]

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Define a simple agent
def simple_agent(state: BasicAgentState) -> BasicAgentState:
    """A simple agent that responds to a message."""
    prompt = f"User asked: {state['message']}. Respond briefly."
    ai_response = llm.invoke(prompt).content
    return {"message": state['message'], "response": ai_response}

# Create workflow
def create_simple_workflow():
    workflow = StateGraph(BasicAgentState)
    workflow.add_node("agent", simple_agent)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow

if __name__ == "__main__":
    simple_wf = create_simple_workflow()

    # Initialize Spectra
    spectra_logger = Spectra(simple_wf)

    # Define initial state
    initial_input = {"message": "Hello, world!"}

    # Run the workflow
    final_output = spectra_logger.run(initial_input)

    print("\nWorkflow Output:")
    print("-----------------")
    print(f"Initial Message: {initial_input['message']}")
    print(f"Agent Response: {final_output['response']}") 