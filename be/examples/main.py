from typing import TypedDict, Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from spectra import Spectra
from dotenv import load_dotenv

load_dotenv()

# Define our state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next_agent: Annotated[str, "The next agent to call"]

# Initialize the LLM
llm = ChatOpenAI(temperature=0.7)

# Create prompt templates for each agent
math_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math expert. You solve math problems and explain your reasoning."),
    ("user", "{input}")
])

explanation_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an explanation expert. You take complex explanations and make them simple and clear."),
    ("user", "{input}")
])

# Define the agents at module level
def math_agent(state: AgentState) -> AgentState:
    """Math expert agent that solves problems and explains reasoning."""
    # Get the last human message
    last_message = state["messages"][-1].content if state["messages"] else "What's the math problem?"
    
    # Create the chain
    chain = math_agent_prompt | llm
    
    # Get the response
    response = chain.invoke({"input": last_message})
    
    # Update state
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "next_agent": "explanation_agent"
    }

def explanation_agent(state: AgentState) -> AgentState:
    """Explanation expert agent that simplifies complex explanations."""
    # Get the last message
    last_message = state["messages"][-1].content
    
    # Create the chain
    chain = explanation_agent_prompt | llm
    
    # Get the response
    response = chain.invoke({"input": f"Please explain this in simpler terms: {last_message}"})
    
    # Update state
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "next_agent": "end"
    }

def should_continue(state: AgentState) -> str:
    """Determine the next node based on the state."""
    return state["next_agent"]

def create_workflow():
    """Create and configure the workflow with all agents and edges."""
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add the nodes
    workflow.add_node("math_agent", math_agent)
    workflow.add_node("explanation_agent", explanation_agent)
    
    # Add the edges
    workflow.add_edge(START, "math_agent")
    workflow.add_conditional_edges(
        "math_agent",
        should_continue,
        {
            "explanation_agent": "explanation_agent",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "explanation_agent",
        should_continue,
        {
            "end": END
        }
    )
    
    return workflow

if __name__ == "__main__":
    # Create the workflow
    workflow = create_workflow()
    
    # Create spectrum logger and run the workflow
    spectra_logger = Spectra(workflow)
    
    # Run the workflow with initial state
    initial_state = {
        "messages": [HumanMessage(content="What is the derivative of x^2?")],
        "next_agent": "math_agent"
    }
    
    # Execute the workflow and get final state
    final_state = spectra_logger.run(initial_state)
    
    # Print the conversation
    print("\nAgent Conversation:")
    print("-----------------")
    for message in final_state["messages"]:
        if isinstance(message, HumanMessage):
            print(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"AI: {message.content}")
        print()
