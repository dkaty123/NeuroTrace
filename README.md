# NeuroTrace: AI Agent Security and Observability Platform

<img width="1490" alt="Screenshot 2025-06-21 at 1 17 09 AM" src="https://github.com/user-attachments/assets/f11447c0-1267-4dc8-b826-328e850f216f" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroTrace isn't just a tool; it's your command center for dissecting, visualizing, and securing complex LangGraph agentic workflows. Dive deep into the execution flow, uncover hidden vulnerabilities, and watch your agents think—all in real-time!**

NeutroTrace provides a dynamic, developer-first experience to bring clarity and control to the often-opaque world of AI agents.

---

## 🚀 Key Features

*   **🔮 Live Agent Code Flow Visualization**: Watch your LangGraph workflows execute node by node on a dynamic graph. Understand connections, data flow, and execution paths at a glance.
*   **🛡️ AI Security Assessment**:
    *   Automatic vulnerability detection in agent source code (SQLi, Command Injection, Hardcoded Secrets, and more!).
    *   Creative vulnerability suggestions powered by GPT-4 for non-obvious risks.
    *   Severity levels (Critical, High, Medium, Low) for prioritized remediation.
*   **📜 Detailed Processed Logs**: A clean, organized view of every event, decision, and hiccup in your workflow execution.
*   **🧠 LLM & Chain Thinking Capture**: (Optional) Peek into the mind of your LLMs! NeuroTrace can log prompts, responses, and chain execution details.
*   **💻 Node Inspection Panel**: Click any node to instantly see its:
    *   Source code with syntax highlighting.
    *   Detailed description and metadata.
    *   Identified vulnerabilities and recommendations.
    *   Option to view code context for each vulnerability.
*   **🌐 Real-time Updates**: The UI polls for new data, keeping your dashboard in sync with the latest workflow executions.
*   **💅 Sleek & Modern UI**: Built with Next.js, TypeScript, and Tailwind CSS for a responsive and visually appealing experience. Includes loading skeletons and smooth transitions.

---

## 🛠️ Tech Stack

*   **Backend**:
    *   Python 3.x
    *   LangGraph: For defining and running agentic workflows.
    *   **NeutroTrace Librar*: Custom Python library to instrument and log LangGraph workflows.
    *   Flask/FastAPI (or similar, if serving Python backend separately - *currently integrated with Next.js API routes*)
*   **Frontend**:
    *   Next.js (React Framework)
    *   TypeScript
    *   Tailwind CSS
    *   ReactFlow: For rendering the dynamic agent flow graph.
    *   Lucide Icons: For a clean icon set.
    *   Framer Motion: For smooth animations.
*   **API**:
    *   Next.js API Routes: For communication between the Neuro library (Python) and the frontend.
*   **Data Storage**:
    *   JSON files (`processed_agent_code.json`, `processed_logs.json`) for simplicity.

---

## ⚡ Getting Started: Unleash the Neuro!

### Prerequisites

*   Node.js (v18.x or later recommended)
*   Python (v3.9 or later recommended)
*   `npm` or `yarn`
*   An OpenAI API Key (if you want to use the AI-powered vulnerability suggestions and log processing features)

### Environment Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <your-repo-url>
    cd agenthacks
    ```

2.  **Set up Environment Variables:**
    Create a `.env.local` file in the `fe` (frontend) directory:
    ```bash
    # fe/.env.local
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key. This is used by the Next.js API routes for interacting with OpenAI.

    For the Python backend (Neuro library), it currently expects the OpenAI key to be available in the environment if any component it uses (like LangChain itself with OpenAI models) requires it. You might need to set this in your shell or a Python-specific environment file if your LangGraph agents directly use OpenAI.
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```

### Installation

1.  **Frontend Dependencies:**
    Navigate to the frontend directory and install the necessary Node.js packages.
    ```bash
    cd fe
    npm install
    # or
    # yarn install
    cd .. 
    ```

2.  **Backend Dependencies:**
    The Neuro library and its examples use Python. It's recommended to use a virtual environment.
    ```bash
    cd be
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    pip install -r requirements.txt # Assuming you have a requirements.txt in be/
    cd ..
    ```
    *Note: If a `requirements.txt` for the `be/` directory doesn't exist, you'll need to create one based on the imports in `neurotrace.py` and your example files (e.g., `langchain`, `requests`, `openai`). A minimal one might include:*
    ```txt
    # be/requirements.txt
    langchain
    langgraph
    requests
    openai
    # Add any other specific versions or libraries your workflow examples need
    ```

---

## 🚀 Running the Beast

You need to run both the frontend (Next.js app) and a Python script that uses the library with your LangGraph workflow.

1.  **Start the Frontend Development Server:**
    ```bash
    cd fe
    npm run dev
    # or
    # yarn dev
    ```
    This will typically start the Next.js application on `http://localhost:3000`. Open this URL in your browser. You should see the dashboard.

2.  **Run a python Workflow:**
    Navigate to the backend directory (where your Python examples are located) and run one of your example scripts that initializes and uses the neurotrace class.
    ```bash
    cd be 
    # Ensure your virtual environment is activated: source venv/bin/activate
    python examples/your_workflow_example.py 
    ```
    *(Replace `examples/your_workflow_example.py` with the actual path to your example script, like `examples/simple_workflow_example.py`)*

    As your Python script runs the LangGraph workflow instrumented:
    *   The `NeuroTrace` library will call the Next.js API endpoints (`/api/process-agent-code`, `/api/process-logs`).
    *   The frontend dashboard will poll these endpoints (`/api/get-agent-code`, `/api/get-processed-logs`) and update in real-time.

**Sequence:**
1.  Start frontend (`npm run dev` in `fe/`).
2.  Run Python workflow script (e.g., `python be/examples/simple_workflow_example.py`).
3.  Observe the NeuroTrace dashboard at `http://localhost:3000` populate with data.

---

## 🤔 How It Works (The Gist)

1.  **Instrumentation :
    *   The class wraps your LangGraph workflow.
    *   During initialization, it inspects all nodes (agent functions) in your graph.
    *   For each node, it extracts source code and metadata.
    *   It sends this agent code data to a Next.js API endpoint (`POST /api/process-agent-code`). This endpoint analyzes the code for vulnerabilities (statically and using OpenAI) and saves it to `processed_agent_code.json`.
    *   It sets up a custom `NTCallbackHandler` for LangChain/LangGraph.
2.  **Real-time Logging & API Calls**:
    *   When you `invoke` or `run` the workflow through the `NeuroTrace` instance:
        *   The `_clear_processed_data` method first calls `DELETE /api/process-agent-code` to reset stored data for a fresh run's perspective.
        *   The `NTCallbackHandler` captures LLM/chain events (prompts, responses, errors, token usage).
        *   These events are logged to a local JSONL file (e.g., in `be/logs/`).
        *   If `auto_process` is enabled, these logs are also sent to `POST /api/process-logs`. This endpoint processes the logs (e.g., with OpenAI for overviews and vulnerability details) and saves them to `processed_logs.json`.
3.  **Frontend Display (`fe/`)**:
    *   The Next.js application provides the dashboard UI.
    *   Pages like `/dash` and `/vulnerabilities` fetch data from:
        *   `GET /api/get-agent-code` (reads `processed_agent_code.json`)
        *   `GET /api/get-processed-logs` (reads `processed_logs.json`)
    *   The data is then rendered into the Agent Code Flow graph, log lists, vulnerability cards, etc.
    *   The frontend polls these GET endpoints periodically to display updates.

---

## 💡 Future Roadmap (Ideas)

*   [ ] **Interactive Log Filtering & Searching**: More advanced controls for sifting through processed logs.
*   [ ] **State Snapshot Viewer**: Visualize the full state object at each step of the workflow.
*   [ ] **Manual Vulnerability Reporting**: Allow users to mark or add custom vulnerabilities.
*   [ ] **Export/Import Functionality**: Save and load workflow analysis sessions.
*   [ ] **Integration with Other Frameworks:** Integrate with other frameworks such as Fetch.ai's uAgents.
---

