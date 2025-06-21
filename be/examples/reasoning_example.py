from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from spectra import Spectra
from dotenv import load_dotenv

load_dotenv()

# Define state
class ReasoningState(TypedDict):
    question: Annotated[str, "The question to answer"]
    thoughts: Annotated[List[str], "The reasoning steps"]
    answer: Annotated[str, "The final answer"]

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Define the reasoning steps
def generate_thoughts(state: ReasoningState) -> ReasoningState:
    """Generate step-by-step reasoning about the question."""
    prompt = f"""
    Question: {state['question']}
    
    Generate 3 logical steps to reason about this question. 
    Think carefully and be thorough.
    """
    response = llm.invoke(prompt).content
    thoughts = [line.strip() for line in response.split('\n') if line.strip()]
    
    return {
        "question": state['question'],
        "thoughts": thoughts,
        "answer": ""  # Will be populated in the next step
    }

def formulate_answer(state: ReasoningState) -> ReasoningState:
    """Formulate a final answer based on the reasoning steps."""
    thoughts_text = '\n'.join([f"- {thought}" for thought in state['thoughts']])
    
    prompt = f"""
    Question: {state['question']}
    
    My reasoning:
    {thoughts_text}
    
    Based on this reasoning, what is the final, concise answer?
    """
    
    answer = llm.invoke(prompt).content
    
    return {
        "question": state['question'],
        "thoughts": state['thoughts'],
        "answer": answer
    }

# Create workflow
def create_reasoning_workflow():
    workflow = StateGraph(ReasoningState)
    
    # Add nodes
    workflow.add_node("generate_thoughts", generate_thoughts)
    workflow.add_node("formulate_answer", formulate_answer)
    
    # Add edges
    workflow.add_edge(START, "generate_thoughts")
    workflow.add_edge("generate_thoughts", "formulate_answer")
    workflow.add_edge("formulate_answer", END)
    
    return workflow

if __name__ == "__main__":
    reasoning_workflow = create_reasoning_workflow()
    
    # Initialize Spectra logger
    spectra_logger = Spectra(reasoning_workflow, output_dir="examples/logs")
    
    # Define the question
    initial_state = {
        "question": "What would happen if the sun suddenly disappeared?",
        "thoughts": [],
        "answer": ""
    }
    
    # Execute the workflow
    final_state = spectra_logger.run(initial_state)
    
    # Print the results
    print("\nReasoning Process:")
    print("-----------------")
    print(f"Question: {final_state['question']}")
    print("\nThoughts:")
    for i, thought in enumerate(final_state['thoughts'], 1):
        print(f"{i}. {thought}")
    
    print("\nFinal Answer:")
    print(f"{final_state['answer']}") 