from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from spectra import Spectra
from dotenv import load_dotenv

load_dotenv()

# Define state
class ThinkingAgentState(TypedDict):
    problem: Annotated[str, "The problem to solve"]
    reasoning_steps: Annotated[list, "Step by step reasoning"]
    solution: Annotated[str, "The final solution"]
    confidence: Annotated[float, "Confidence in the solution"]

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

def reasoning_agent(state: ThinkingAgentState) -> ThinkingAgentState:
    """Agent that breaks down problems into reasoning steps."""
    
    reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a logical reasoning expert. Break down the given problem into clear, numbered steps.
        Think step by step and show your reasoning process.
        
        Format your response as:
        1. [First step]
        2. [Second step]
        3. [etc...]
        
        Be thorough and explain each step clearly."""),
        ("user", "Problem: {problem}")
    ])
    
    chain = reasoning_prompt | llm
    response = chain.invoke({"problem": state["problem"]})
    
    # Extract steps (simple parsing)
    steps = []
    for line in response.content.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            steps.append(line)
    
    return {
        "problem": state["problem"],
        "reasoning_steps": steps,
        "solution": state.get("solution", ""),
        "confidence": state.get("confidence", 0.0)
    }

def solution_agent(state: ThinkingAgentState) -> ThinkingAgentState:
    """Agent that provides the final solution based on reasoning steps."""
    
    solution_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a solution expert. Based on the reasoning steps provided, 
        give a clear, concise final answer. Also rate your confidence from 0.0 to 1.0."""),
        ("user", """Problem: {problem}
        
Reasoning Steps:
{reasoning_steps}

Provide:
1. A clear final answer
2. Your confidence level (0.0 to 1.0)

Format: 
Answer: [your answer]
Confidence: [0.0-1.0]""")
    ])
    
    steps_text = '\n'.join(state["reasoning_steps"])
    chain = solution_prompt | llm
    response = chain.invoke({
        "problem": state["problem"],
        "reasoning_steps": steps_text
    })
    
    # Simple parsing to extract answer and confidence
    content = response.content
    solution = "Unable to determine"
    confidence = 0.5
    
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Answer:'):
            solution = line.replace('Answer:', '').strip()
        elif line.startswith('Confidence:'):
            try:
                confidence = float(line.replace('Confidence:', '').strip())
            except:
                confidence = 0.5
    
    return {
        "problem": state["problem"],
        "reasoning_steps": state["reasoning_steps"],
        "solution": solution,
        "confidence": confidence
    }

def create_thinking_workflow():
    """Create a workflow that demonstrates LLM thinking capture."""
    workflow = StateGraph(ThinkingAgentState)
    
    # Add nodes
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("solution", solution_agent)
    
    # Add edges
    workflow.add_edge(START, "reasoning")
    workflow.add_edge("reasoning", "solution")
    workflow.add_edge("solution", END)
    
    return workflow

if __name__ == "__main__":
    print("Running Thinking Capture Example")
    print("=" * 40)
    print("This example captures LLM 'thinking' - prompts, responses, and token usage")
    print()
    
    # Create workflow
    workflow = create_thinking_workflow()
    
    # Initialize Spectra with thinking capture enabled
    spectra_logger = Spectra(workflow, capture_thinking=True)
    
    # Test problems that require reasoning
    test_problems = [
        "If a train leaves New York at 3 PM traveling at 60 mph, and another train leaves Chicago at 4 PM traveling at 80 mph toward New York, and the cities are 800 miles apart, when will they meet?",
        "A farmer has chickens and cows. The animals have 74 legs total and 26 heads. How many chickens and how many cows are there?",
        "What is the next number in this sequence: 2, 6, 12, 20, 30, ?"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🧠 Problem {i}:")
        print(f"'{problem}'")
        print("\nProcessing...")
        
        initial_state = {
            "problem": problem,
            "reasoning_steps": [],
            "solution": "",
            "confidence": 0.0
        }
        
        # Run workflow (thinking will be captured in JSONL)
        final_state = spectra_logger.run(initial_state)
        
        print(f"\n📝 Reasoning Steps:")
        for step in final_state["reasoning_steps"]:
            print(f"  {step}")
        
        print(f"\n✅ Solution: {final_state['solution']}")
        print(f"🎯 Confidence: {final_state['confidence']:.1%}")
        print("-" * 60)
    
    print(f"\n💾 Check the logs for detailed LLM thinking capture!")
    print(f"   - JSON: Workflow structure and functions")
    print(f"   - JSONL: Live execution with prompts, responses, and token usage") 