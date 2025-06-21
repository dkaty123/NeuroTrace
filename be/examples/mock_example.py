from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from spectra import Spectra

# Define state
class MockAgentState(TypedDict):
    input_text: Annotated[str, "The input text"]
    processed_text: Annotated[str, "The processed text"]
    step_count: Annotated[int, "Number of processing steps"]

def preprocessor(state: MockAgentState) -> MockAgentState:
    """Preprocesses the input text by converting to uppercase."""
    processed = state['input_text'].upper()
    return {
        "input_text": state['input_text'],
        "processed_text": processed,
        "step_count": state.get('step_count', 0) + 1
    }

def word_counter(state: MockAgentState) -> MockAgentState:
    """Counts words and adds info to processed text."""
    word_count = len(state['processed_text'].split())
    processed = f"{state['processed_text']} [WORDS: {word_count}]"
    return {
        "input_text": state['input_text'],
        "processed_text": processed,
        "step_count": state['step_count'] + 1
    }

def finalizer(state: MockAgentState) -> MockAgentState:
    """Adds final processing info."""
    processed = f"{state['processed_text']} [STEPS: {state['step_count'] + 1}]"
    return {
        "input_text": state['input_text'],
        "processed_text": processed,
        "step_count": state['step_count'] + 1
    }

def create_mock_workflow():
    """Create a simple workflow that doesn't need API keys."""
    workflow = StateGraph(MockAgentState)
    
    # Add nodes
    workflow.add_node("preprocess", preprocessor)
    workflow.add_node("count_words", word_counter)
    workflow.add_node("finalize", finalizer)
    
    # Add edges
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "count_words")
    workflow.add_edge("count_words", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow

if __name__ == "__main__":
    print("Running Mock Spectra Example (No API Keys Required)")
    print("=" * 50)
    
    # Create workflow
    workflow = create_mock_workflow()
    
    # Initialize Spectra (logs will go to ./logs/ by default)
    # Disable thinking capture for this simple example
    spectra_logger = Spectra(workflow, capture_thinking=False)
    
    # Test with different inputs
    test_inputs = [
        "hello world this is a test",
        "spectra is awesome for logging workflows",
        "short text"
    ]
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\nTest {i}:")
        print(f"Input: {input_text}")
        
        initial_state = {
            "input_text": input_text,
            "processed_text": "",
            "step_count": 0
        }
        
        # Run workflow
        final_state = spectra_logger.run(initial_state)
        
        print(f"Output: {final_state['processed_text']}")
        print(f"Total Steps: {final_state['step_count']}") 