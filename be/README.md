# Spectra

A Python library for logging and analyzing LangGraph workflows.

## About

Spectra is a utility library for LangGraph that allows you to:
- Log workflow node functions and their source code
- Track state types through workflow execution
- Organize and execute workflows with a simple API

## Installation

### Quick Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd spectra/be

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Spectra in editable mode with all dependencies
pip install -e .
```

### For OpenAI Examples

If you want to run the AI-powered examples, create a `.env` file:

```bash
# Create .env file in the be/ directory
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

## Examples

The `examples` directory contains several demos:

### 1. Mock Example (No API keys needed)
```bash
python examples/mock_example.py
```
Demonstrates Spectra's logging capabilities with a simple text processing workflow.

### 2. Advanced Research Assistant (Requires API key) ⭐
```bash
python examples/research_assistant.py
```
**Sophisticated multi-agent workflow featuring:**
- 6 specialized agents (Query Analyzer, Web Researcher, Content Analyzer, Synthesis Agent, Fact Checker, Report Generator)
- Complex state management with 15+ state fields
- Conditional routing with error handling and retry logic
- Tool integration (simulated web search)
- Multiple LLM models for different tasks (GPT-4 for analysis, GPT-3.5-turbo for research)
- Rich LLM thinking capture showing detailed agent interactions

### 3. LLM Thinking Capture (Requires API key)
```bash
python examples/thinking_example.py
```
Shows how Spectra captures LLM "thinking" - prompts, responses, token usage, and reasoning chains.

### 4. Other OpenAI Examples (Requires API key)
```bash
python examples/simple_workflow_example.py  # Single agent
python examples/main.py                     # Multi-agent workflow  
python examples/reasoning_example.py        # Step-by-step reasoning
```

## How It Works

Spectra analyzes your LangGraph workflow, extracts metadata like state types and node functions, and logs this information:

```python
from spectra import Spectra
from langgraph.graph import StateGraph

# Create your workflow
workflow = StateGraph(YourStateType)
# ... add nodes and edges ...

# Initialize Spectra with LLM thinking capture
spectra_logger = Spectra(workflow, capture_thinking=True)

# Run the workflow
final_state = spectra_logger.run(initial_state)
```

### Output

Spectra creates detailed logs in multiple formats:

**JSON Files** (for analysis):
- `node_functions_YYYYMMDD_HHMMSS.json` - Source code and metadata of all node functions
- Contains state type information, execution metadata, and timestamps

**JSONL Files** (for live monitoring):
- `workflow_execution_YYYYMMDD_HHMMSS.jsonl` - Live execution events
- Real-time logging of workflow initialization, compilation, execution start/end
- **LLM Thinking Capture**: Prompts sent to LLMs, responses received, token usage, and chain processing
- Each line is a JSON object with timestamp and event data

All logs are stored in the `be/logs/` directory by default.

### LLM Thinking Capture

When `capture_thinking=True` (default), Spectra logs:
- **Prompts** sent to language models
- **Responses** received from LLMs  
- **Token usage** and model information
- **Chain processing** steps and intermediate outputs

This gives you complete visibility into what your agents are "thinking" during execution.

## Development

To contribute to Spectra:

1. Clone the repository
2. Follow the installation steps above
3. Make your changes to the `spectra/` package
4. Test with the provided examples
5. Add new examples for new features

## Requirements

- Python 3.12+
- LangGraph 0.4.5+
- LangChain dependencies (automatically installed)
