"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Book, PlayCircle, Download, Code, Eye, Shield, Terminal, Sparkles, 
  ChevronRight, Copy, Check, ExternalLink, Github, ArrowRight,
  Activity, Database, Zap, Lock, Layers, Search, Calendar,
  FileCode, Cpu, Gauge, AlertTriangle, CheckCircle, ArrowLeft
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Link from "next/link"
import Image from "next/image"

// Code Block Component with syntax highlighting
const CodeBlock = ({ 
  children, 
  language = "python", 
  filename, 
  copyable = true 
}: { 
  children: string
  language?: string
  filename?: string
  copyable?: boolean
}) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(children)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const highlightPython = (code: string) => {
    const tokenColors = {
      comment: 'text-gray-500',
      string: 'text-emerald-400',
      keyword: 'text-purple-400',
      neurotrace: 'text-blue-400',
      library: 'text-teal-400',
      builtin: 'text-sky-400',
      funcName: 'text-yellow-400',
      decorator: 'text-pink-400',
      number: 'text-orange-400',
      default: 'text-gray-300',
    }

    let highlighted = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')

    // Comments
    highlighted = highlighted.replace(
      /(#.*$)/gm,
      `<span class="${tokenColors.comment}">$1</span>`
    )

    // Strings
    highlighted = highlighted.replace(
      /(".*?"|'.*?')/g,
      `<span class="${tokenColors.string}">$1</span>`
    )

    // Keywords
    highlighted = highlighted.replace(
      /\b(from|import|def|return|class|as|with|True|False|None|if|else|elif|for|while|try|except|finally|lambda|yield|pass|break|continue|global|nonlocal|in|is|not|and|or)\b/g,
      `<span class="${tokenColors.keyword}">$1</span>`
    )

    // NeuroTrace
    highlighted = highlighted.replace(
      /\b(NeuroTrace)\b/g,
      `<span class="${tokenColors.neurotrace}">$1</span>`
    )

    // Libraries
    highlighted = highlighted.replace(
      /\b(StateGraph|LLMChain|ChatOpenAI)\b/g,
      `<span class="${tokenColors.library}">$1</span>`
    )

    // Numbers
    highlighted = highlighted.replace(
      /\b(\d+\.?\d*)\b/g,
      `<span class="${tokenColors.number}">$1</span>`
    )

    return highlighted
  }

  return (
    <div className="relative group">
      {filename && (
        <div className="flex items-center justify-between px-4 py-2 bg-zinc-800 border-b border-zinc-700 rounded-t-lg">
          <span className="text-sm text-zinc-300 font-mono">{filename}</span>
          <Badge variant="outline" className="text-xs border-zinc-600 text-zinc-400">
            {language}
          </Badge>
        </div>
      )}
      <div className="relative bg-zinc-900 rounded-lg overflow-hidden border border-zinc-700">
        {copyable && (
          <button
            onClick={handleCopy}
            className="absolute top-3 right-3 p-2 bg-zinc-800 hover:bg-zinc-700 rounded-md opacity-0 group-hover:opacity-100 transition-opacity border border-zinc-600"
          >
            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4 text-zinc-400" />}
          </button>
        )}
        <pre className="p-4 text-sm overflow-x-auto">
          <code 
            dangerouslySetInnerHTML={{ 
              __html: language === 'python' ? highlightPython(children) : children 
            }}
            className="text-zinc-300"
          />
        </pre>
      </div>
    </div>
  )
}

// Feature Card Component
const FeatureCard = ({ icon: Icon, title, description, gradient }: {
  icon: any
  title: string
  description: string
  gradient: string
}) => (
  <motion.div
    whileHover={{ scale: 1.02 }}
    className={`relative p-6 rounded-xl border border-zinc-700/50 bg-gradient-to-br ${gradient} backdrop-blur-sm`}
  >
    <div className="flex items-start gap-4">
      <div className="p-2 bg-white/10 rounded-lg">
        <Icon className="w-6 h-6" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-white mb-2">{title}</h3>
        <p className="text-zinc-300 text-sm leading-relaxed">{description}</p>
      </div>
    </div>
  </motion.div>
)

// Navigation Component
const DocsNavigation = ({ activeSection, setActiveSection }: {
  activeSection: string
  setActiveSection: (section: string) => void
}) => {
  const navigation = [
    { id: "introduction", label: "Introduction", icon: Book },
    { id: "installation", label: "Installation", icon: Download },
    { id: "quickstart", label: "Quick Start", icon: PlayCircle },
    { id: "examples", label: "Examples", icon: Code },
    { id: "monitoring", label: "Monitoring", icon: Eye },
    { id: "security", label: "Security Analysis", icon: Shield },
    { id: "api", label: "API Reference", icon: Terminal },
    { id: "advanced", label: "Advanced", icon: Sparkles },
  ]

  return (
    <div className="space-y-6">
      {/* Back to Dashboard */}
      <Link href="/dashboard">
        <Button variant="outline" className="w-full justify-start border-zinc-600 text-zinc-300 hover:bg-zinc-800">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard
        </Button>
      </Link>

      {/* Logo Section */}
      <div className="flex items-center gap-3 px-3">
        <div className="flex items-center">
          <Image
            src="/neurotrace-mark.svg"
            alt="NeuroTrace Logo"
            width={32}
            height={32}
            className="invert transition-transform duration-300 group-hover:scale-110"
          />
        </div>
        <div>
          <h2 className="text-lg font-bold text-white">NeuroTrace Docs</h2>
          <p className="text-xs text-zinc-400">v1.0.0</p>
        </div>
      </div>

      {/* Navigation */}
      <div className="space-y-2">
        {navigation.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveSection(item.id)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all duration-200 ${
              activeSection === item.id
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-lg shadow-purple-500/10'
                : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50 border border-transparent'
            }`}
          >
            <item.icon className="w-4 h-4" />
            {item.label}
            {activeSection === item.id && (
              <motion.div
                layoutId="activeIndicator"
                className="ml-auto"
              >
                <ChevronRight className="w-4 h-4" />
              </motion.div>
            )}
          </button>
        ))}
      </div>

      {/* Quick Links */}
      <div className="pt-4 border-t border-zinc-700/50">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">Quick Links</h3>
        <div className="space-y-2">
          <Button variant="ghost" size="sm" className="w-full justify-start text-zinc-400 hover:text-white">
            <Github className="w-4 h-4 mr-2" />
            GitHub
          </Button>
          <Button variant="ghost" size="sm" className="w-full justify-start text-zinc-400 hover:text-white">
            <ExternalLink className="w-4 h-4 mr-2" />
            API Status
          </Button>
        </div>
      </div>
    </div>
  )
}

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState("introduction")

  const renderContent = () => {
    switch (activeSection) {
      case "introduction":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-12"
          >
            {/* Hero Section */}
            <div className="space-y-6">
              <motion.div
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30 px-4 py-2">
                  Python 3.12+ - Latest Version
                </Badge>
              </motion.div>
              
              <h1 className="text-4xl font-bold bg-gradient-to-r from-white via-purple-200 to-cyan-200 bg-clip-text text-transparent">
                NeuroTrace Documentation
              </h1>
              
              <p className="text-xl text-zinc-300 max-w-3xl leading-relaxed">
                A Python library for logging and analyzing LangGraph workflows. Track workflow execution, 
                capture LLM thinking, and organize your agent workflows with comprehensive observability.
              </p>
              
              <div className="flex gap-4">
                <Button 
                  onClick={() => setActiveSection("quickstart")}
                  className="bg-purple-600 hover:bg-purple-700 px-6 py-3"
                >
                  Get Started <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
                <Button variant="outline" className="px-6 py-3 border-zinc-600 text-zinc-300 hover:bg-zinc-800">
                  <Github className="w-4 h-4 mr-2" />
                  View on GitHub
                </Button>
              </div>
            </div>

            {/* Feature Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <FeatureCard
                icon={Activity}
                title="Workflow Logging"
                description="Log workflow node functions, source code, and track state types through workflow execution with detailed metadata."
                gradient="from-purple-500/10 to-violet-500/10"
              />
              <FeatureCard
                icon={Eye}
                title="LLM Thinking Capture"
                description="Capture prompts sent to LLMs, responses received, token usage, and complete reasoning chains for full visibility."
                gradient="from-emerald-500/10 to-teal-500/10"
              />
              <FeatureCard
                icon={Database}
                title="Multiple Output Formats"
                description="JSON files for analysis and JSONL files for live monitoring with timestamps and structured event data."
                gradient="from-cyan-500/10 to-blue-500/10"
              />
              <FeatureCard
                icon={Zap}
                title="Simple API"
                description="Organize and execute workflows with a simple, intuitive API. Just wrap your LangGraph workflow with NeuroTrace."
                gradient="from-yellow-500/10 to-orange-500/10"
              />
              <FeatureCard
                icon={Shield}
                title="State Type Tracking"
                description="Automatically track state types through workflow execution with comprehensive metadata collection."
                gradient="from-indigo-500/10 to-purple-500/10"
              />
              <FeatureCard
                icon={Layers}
                title="LangGraph Integration"
                description="Native support for LangGraph workflows with automatic node function extraction and state analysis."
                gradient="from-pink-500/10 to-rose-500/10"
              />
            </div>

            {/* Quick Example */}
            <div className="space-y-6">
              <h2 className="text-3xl font-bold text-white">Quick Example</h2>
              <p className="text-lg text-zinc-300">
                Getting started with NeuroTrace is simple. Just wrap your LangGraph workflow:
              </p>
              
              <CodeBlock filename="simple_workflow.py">
{`from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from neurotrace import NeuroTrace
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
workflow = StateGraph(BasicAgentState)
workflow.add_node("agent", simple_agent)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Initialize NeuroTrace with LLM thinking capture
neurotrace_logger = NeuroTrace(workflow, capture_thinking=True)

# Run the workflow
final_output = neurotrace_logger.run({"message": "Hello, world!"})

print(f"Agent Response: {final_output['response']}")`}
              </CodeBlock>

              <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-lg p-6">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-blue-500/20 rounded-lg">
                    <CheckCircle className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-white mb-2">What NeuroTrace Logs</h4>
                    <p className="text-blue-200 mb-3">
                      NeuroTrace creates detailed logs in the `be/logs/` directory:
                    </p>
                    <ul className="text-blue-200 space-y-1 text-sm">
                      <li>• <strong>JSON files</strong> - Source code and metadata of node functions</li>
                      <li>• <strong>JSONL files</strong> - Live execution events with timestamps</li>
                      <li>• <strong>LLM thinking</strong> - Prompts, responses, and token usage</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Next Steps */}
            <div className="space-y-6">
              <h2 className="text-3xl font-bold text-white">Next Steps</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card 
                  className="bg-gradient-to-br from-purple-500/10 to-violet-500/10 border-purple-500/30 cursor-pointer hover:border-purple-400/50 transition-all hover:shadow-lg hover:shadow-purple-500/10"
                  onClick={() => setActiveSection("installation")}
                >
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-semibold text-white mb-2">Installation</h3>
                        <p className="text-zinc-300">Clone the repo and set up your development environment</p>
                      </div>
                      <ArrowRight className="w-5 h-5 text-purple-400" />
                    </div>
                  </CardContent>
                </Card>
                
                <Card 
                  className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-500/30 cursor-pointer hover:border-cyan-400/50 transition-all hover:shadow-lg hover:shadow-cyan-500/10"
                  onClick={() => setActiveSection("examples")}
                >
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-semibold text-white mb-2">Examples</h3>
                        <p className="text-zinc-300">Explore ready-to-run examples and advanced workflows</p>
                      </div>
                      <ArrowRight className="w-5 h-5 text-cyan-400" />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </motion.div>
        )

      case "installation":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">Installation</h1>
              <p className="text-lg text-zinc-300 mb-8">
                Quick setup guide for getting NeuroTrace running in your environment.
              </p>
            </div>

            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-semibold text-white mb-4">Quick Setup</h2>
                <CodeBlock filename="terminal" language="bash">
{`# Clone the repository
git clone <your-repo-url>
cd neurotrace/be

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install NeuroTrace in editable mode with all dependencies
pip install -e .`}
                </CodeBlock>
              </div>

              <div>
                <h2 className="text-2xl font-semibold text-white mb-4">Environment Setup</h2>
                <p className="text-zinc-300 mb-4">
                  For OpenAI-powered examples, create a `.env` file in the `be/` directory:
                </p>
                <CodeBlock filename=".env">
{`# Create .env file in the be/ directory
OPENAI_API_KEY=your_openai_api_key_here`}
                </CodeBlock>
              </div>

              <div>
                <h2 className="text-2xl font-semibold text-white mb-4">System Requirements</h2>
                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardContent className="p-6">
                    <ul className="space-y-2 text-zinc-300">
                      <li>• <strong>LangGraph 0.4.5+</strong> - Core workflow framework</li>
                      <li>• <strong>LangChain dependencies</strong> - Automatically installed</li>
                      <li>• <strong>Python 3.12+</strong> - Latest Python version required</li>
                      <li>• <strong>OpenAI API key</strong> - Optional, for AI-powered examples</li>
                    </ul>
                  </CardContent>
                </Card>
              </div>

              <div>
                <h2 className="text-2xl font-semibold text-white mb-4">Verify Installation</h2>
                <p className="text-zinc-300 mb-4">Test your installation with the mock example:</p>
                <CodeBlock filename="terminal" language="bash">
{`# Test without API keys
python examples/mock_example.py

# Should output workflow processing steps
# All logs will be saved to be/logs/ directory`}
                </CodeBlock>
              </div>
            </div>

            <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 rounded-lg p-6">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-emerald-400 mt-1" />
                <div>
                  <h4 className="text-lg font-semibold text-white mb-2">Installation Complete!</h4>
                  <p className="text-emerald-200">
                    You're ready to start logging LangGraph workflows. Check out the examples to see NeuroTrace in action.
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case "quickstart":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">Quick Start</h1>
              <p className="text-lg text-zinc-300 mb-8">
                Get your first NeuroTrace-instrumented workflow running in under 5 minutes.
              </p>
            </div>

            <div className="space-y-8">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">1</div>
                  <h3 className="text-xl font-semibold text-white">Install NeuroTrace</h3>
                </div>
                <p className="text-zinc-300 ml-11">First, set up your environment:</p>
                <div className="ml-11">
                  <CodeBlock filename="terminal" language="bash">
{`cd neurotrace/be
python3 -m venv .venv
source .venv/bin/activate
pip install -e .`}
                  </CodeBlock>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">2</div>
                  <h3 className="text-xl font-semibold text-white">Try the Mock Example</h3>
                </div>
                <p className="text-zinc-300 ml-11">Run a simple example that doesn't require API keys:</p>
                <div className="ml-11">
                  <CodeBlock filename="mock_example.py">
{`from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from neurotrace import NeuroTrace

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

# Create workflow
workflow = StateGraph(MockAgentState)
workflow.add_node("preprocess", preprocessor)
workflow.add_node("count_words", word_counter)
workflow.add_edge(START, "preprocess")
workflow.add_edge("preprocess", "count_words")
workflow.add_edge("count_words", END)

# Initialize NeuroTrace (logs will go to ./logs/ by default)
neurotrace_logger = NeuroTrace(workflow, capture_thinking=False)

# Run workflow
initial_state = {
    "input_text": "hello world this is a test",
    "processed_text": "",
    "step_count": 0
}

final_state = neurotrace_logger.run(initial_state)
print(f"Output: {final_state['processed_text']}")`}
                  </CodeBlock>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">3</div>
                  <h3 className="text-xl font-semibold text-white">Run and Check Logs</h3>
                </div>
                <p className="text-zinc-300 ml-11">Execute the example and explore the generated logs:</p>
                <div className="ml-11">
                  <CodeBlock filename="terminal" language="bash">
{`python examples/mock_example.py

# Output:
# Running Mock NeuroTrace Example (No API Keys Required)
# Test 1: hello world this is a test
# Output: HELLO WORLD THIS IS A TEST [WORDS: 5] [STEPS: 3]

# Check the logs directory
ls be/logs/
# node_functions_YYYYMMDD_HHMMSS.json
# workflow_execution_YYYYMMDD_HHMMSS.jsonl`}
                  </CodeBlock>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">4</div>
                  <h3 className="text-xl font-semibold text-white">Try LLM Examples (Optional)</h3>
                </div>
                <p className="text-zinc-300 ml-11">For AI-powered examples, add your OpenAI API key:</p>
                <div className="ml-11">
                  <CodeBlock filename="terminal" language="bash">
{`# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Run AI-powered examples
python examples/simple_workflow_example.py
python examples/thinking_example.py
python examples/research_assistant.py  # Advanced multi-agent workflow`}
                  </CodeBlock>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/30">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-3">
                    <Database className="w-6 h-6 text-green-400" />
                    <h4 className="text-lg font-semibold text-white">JSON Logging</h4>
                  </div>
                  <p className="text-green-200 text-sm">
                    Node functions and metadata stored in structured JSON files for analysis.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-3">
                    <Activity className="w-6 h-6 text-blue-400" />
                    <h4 className="text-lg font-semibold text-white">Live Monitoring</h4>
                  </div>
                  <p className="text-blue-200 text-sm">
                    Real-time JSONL logs with timestamps for live workflow monitoring.
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-purple-500/10 to-violet-500/10 border-purple-500/30">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-3">
                    <Eye className="w-6 h-6 text-purple-400" />
                    <h4 className="text-lg font-semibold text-white">LLM Thinking</h4>
                  </div>
                  <p className="text-purple-200 text-sm">
                    Capture prompts, responses, and reasoning chains from language models.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 rounded-lg p-6">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-emerald-400 mt-1" />
                <div>
                  <h4 className="text-lg font-semibold text-white mb-2">What's Next?</h4>
                  <p className="text-emerald-200 mb-4">
                    Your workflow is now instrumented! Explore the logs directory to see:
                  </p>
                  <ul className="text-emerald-200 space-y-1">
                    <li>• JSON files with source code and metadata</li>
                    <li>• JSONL files with execution events and timestamps</li>
                    <li>• LLM thinking capture (when enabled)</li>
                    <li>• State type tracking and workflow analysis</li>
                  </ul>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case "examples":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">Examples</h1>
              <p className="text-lg text-zinc-300 mb-8">
                The examples directory contains several demos showcasing different NeuroTrace capabilities.
              </p>
            </div>

            <div className="space-y-12">
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">1. Mock Example (No API Keys Needed)</h2>
                <p className="text-zinc-300">
                  Demonstrates NeuroTrace's logging capabilities with a simple text processing workflow.
                </p>
                
                <CodeBlock filename="examples/mock_example.py">
{`python examples/mock_example.py`}
                </CodeBlock>

                <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-white mb-3">Features Demonstrated</h4>
                  <ul className="text-blue-200 space-y-1 text-sm">
                    <li>• Simple text processing workflow without LLM dependencies</li>
                    <li>• State type tracking with annotated fields</li>
                    <li>• Step counting and processing chain logging</li>
                    <li>• Basic NeuroTrace integration with `capture_thinking=False`</li>
                  </ul>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">2. Advanced Research Assistant ⭐ (Requires API Key)</h2>
                <p className="text-zinc-300">
                  Sophisticated multi-agent workflow featuring complex state management and conditional routing.
                </p>
                
                <CodeBlock filename="examples/research_assistant.py">
{`python examples/research_assistant.py`}
                </CodeBlock>

                <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-white mb-3">Advanced Features</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-emerald-200 text-sm">
                    <div>
                      <h5 className="font-semibold text-emerald-100 mb-2">Workflow Architecture:</h5>
                      <ul className="space-y-1">
                        <li>• 6 specialized agents</li>
                        <li>• Complex state with 15+ fields</li>
                        <li>• Conditional routing & error handling</li>
                        <li>• Tool integration (simulated web search)</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-semibold text-emerald-100 mb-2">LLM Integration:</h5>
                      <ul className="space-y-1">
                        <li>• Multiple LLM models (GPT-4, GPT-3.5-turbo)</li>
                        <li>• Rich LLM thinking capture</li>
                        <li>• Detailed agent interactions</li>
                        <li>• Retry logic and error handling</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">3. LLM Thinking Capture (Requires API Key)</h2>
                <p className="text-zinc-300">
                  Shows how NeuroTrace captures LLM "thinking" - prompts, responses, token usage, and reasoning chains.
                </p>
                
                <CodeBlock filename="examples/thinking_example.py">
{`python examples/thinking_example.py`}
                </CodeBlock>

                <div className="bg-gradient-to-r from-purple-500/10 to-violet-500/10 border border-purple-500/20 rounded-lg p-6">
                  <h4 className="text-lg font-semibold text-white mb-3">LLM Monitoring Features</h4>
                  <ul className="text-purple-200 space-y-1 text-sm">
                    <li>• <strong>Prompts</strong> - Complete prompts sent to language models</li>
                    <li>• <strong>Responses</strong> - Full responses received from LLMs</li>
                    <li>• <strong>Token Usage</strong> - Input/output tokens and cost tracking</li>
                    <li>• <strong>Chain Processing</strong> - Step-by-step reasoning capture</li>
                  </ul>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">4. Other OpenAI Examples</h2>
                <p className="text-zinc-300">Additional examples showcasing different workflow patterns.</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardContent className="p-6">
                      <h4 className="text-lg font-semibold text-white mb-3">Simple Workflow</h4>
                      <CodeBlock>
{`python examples/simple_workflow_example.py`}
                      </CodeBlock>
                      <p className="text-zinc-400 text-sm mt-2">Single agent with basic LLM integration</p>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardContent className="p-6">
                      <h4 className="text-lg font-semibold text-white mb-3">Multi-Agent Workflow</h4>
                      <CodeBlock>
{`python examples/main.py`}
                      </CodeBlock>
                      <p className="text-zinc-400 text-sm mt-2">Multiple agents working together</p>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/50 border-zinc-700 md:col-span-2">
                    <CardContent className="p-6">
                      <h4 className="text-lg font-semibold text-white mb-3">Reasoning Example</h4>
                      <CodeBlock>
{`python examples/reasoning_example.py`}
                      </CodeBlock>
                      <p className="text-zinc-400 text-sm mt-2">Step-by-step reasoning and decision making</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>

            <div className="mt-8">
              <Card className="bg-gradient-to-r from-amber-500/10 to-orange-500/10 border-amber-500/30">
                <CardContent className="p-6">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-6 h-6 text-amber-400 mt-1" />
                    <div>
                      <h4 className="text-lg font-semibold text-white mb-2">API Key Setup</h4>
                      <p className="text-amber-200 mb-4">
                        For OpenAI-powered examples, create a `.env` file in the `be/` directory:
                      </p>
                      <CodeBlock>
{`echo "OPENAI_API_KEY=your_openai_api_key_here" > .env`}
                      </CodeBlock>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )

      case "monitoring":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">Monitoring & Observability</h1>
              <p className="text-lg text-zinc-300 mb-8">
                Understand how NeuroTrace provides deep insights into your LangGraph workflows.
              </p>
            </div>

            <div className="space-y-12">
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">What Gets Monitored</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <FileCode className="w-6 h-6 text-blue-400" />
                        <h3 className="text-lg font-semibold text-white">Workflow Structure</h3>
                      </div>
                      <ul className="text-blue-200 space-y-2 text-sm">
                        <li>• Node function source code extraction</li>
                        <li>• State type information and annotations</li>
                        <li>• Workflow graph structure and edges</li>
                        <li>• Execution metadata and timestamps</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-purple-500/10 to-violet-500/10 border-purple-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Activity className="w-6 h-6 text-purple-400" />
                        <h3 className="text-lg font-semibold text-white">Runtime Execution</h3>
                      </div>
                      <ul className="text-purple-200 space-y-2 text-sm">
                        <li>• Real-time workflow execution events</li>
                        <li>• Node-by-node state transitions</li>
                        <li>• Processing steps and flow control</li>
                        <li>• Error handling and exception tracking</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border-emerald-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Eye className="w-6 h-6 text-emerald-400" />
                        <h3 className="text-lg font-semibold text-white">LLM Interactions</h3>
                      </div>
                      <ul className="text-emerald-200 space-y-2 text-sm">
                        <li>• Complete prompts sent to models</li>
                        <li>• Full responses from language models</li>
                        <li>• Token usage and cost tracking</li>
                        <li>• Model information and parameters</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-amber-500/10 to-orange-500/10 border-amber-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Gauge className="w-6 h-6 text-amber-400" />
                        <h3 className="text-lg font-semibold text-white">Performance Metrics</h3>
                      </div>
                      <ul className="text-amber-200 space-y-2 text-sm">
                        <li>• Execution timing and latency</li>
                        <li>• Memory usage patterns</li>
                        <li>• Chain processing performance</li>
                        <li>• Workflow compilation metrics</li>
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Log File Structure</h2>
                <p className="text-zinc-300">NeuroTrace generates two types of log files for different analysis needs:</p>

                <Tabs defaultValue="json" className="space-y-6">
                  <TabsList className="grid grid-cols-2 w-full max-w-md bg-zinc-800 border border-zinc-700">
                    <TabsTrigger value="json" className="data-[state=active]:bg-purple-600">JSON Files</TabsTrigger>
                    <TabsTrigger value="jsonl" className="data-[state=active]:bg-purple-600">JSONL Files</TabsTrigger>
                  </TabsList>

                  <TabsContent value="json" className="space-y-6">
                    <Card className="bg-zinc-900/50 border-zinc-700">
                      <CardHeader>
                        <CardTitle className="text-white">node_functions_YYYYMMDD_HHMMSS.json</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-zinc-300 mb-4">Structured analysis data containing:</p>
                        <CodeBlock>
{`{
  "workflow_metadata": {
    "timestamp": "2024-01-15T10:30:45Z",
    "workflow_type": "StateGraph",
    "state_schema": "BasicAgentState"
  },
  "node_functions": [
    {
      "node_name": "simple_agent",
      "function_name": "simple_agent",
      "source_code": "def simple_agent(state: BasicAgentState)...",
      "parameters": [...],
      "return_type": "BasicAgentState"
    }
  ],
  "state_types": {
    "BasicAgentState": {
      "fields": {...},
      "annotations": {...}
    }
  }
}`}
                        </CodeBlock>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="jsonl" className="space-y-6">
                    <Card className="bg-zinc-900/50 border-zinc-700">
                      <CardHeader>
                        <CardTitle className="text-white">workflow_execution_YYYYMMDD_HHMMSS.jsonl</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-zinc-300 mb-4">Real-time execution events, one JSON object per line:</p>
                        <CodeBlock>
{`{"timestamp": "2024-01-15T10:30:45.123Z", "event": "workflow_init", "data": {...}}
{"timestamp": "2024-01-15T10:30:45.456Z", "event": "node_start", "node": "agent", "state": {...}}
{"timestamp": "2024-01-15T10:30:46.789Z", "event": "llm_prompt", "prompt": "User asked...", "tokens": 15}
{"timestamp": "2024-01-15T10:30:47.012Z", "event": "llm_response", "response": "The capital...", "tokens": 8}
{"timestamp": "2024-01-15T10:30:47.234Z", "event": "node_complete", "node": "agent", "result": {...}}`}
                        </CodeBlock>
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Monitoring Best Practices</h2>
                
                <div className="space-y-4">
                  <Card className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-green-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-green-400 mt-1" />
                        <div>
                          <h4 className="text-lg font-semibold text-white mb-2">Enable LLM Thinking Capture</h4>
                          <p className="text-green-200 mb-3">
                            Set <code className="text-green-300">capture_thinking=True</code> for comprehensive observability:
                          </p>
                          <CodeBlock>
{`# Enable full LLM monitoring
neurotrace_logger = NeuroTrace(workflow, capture_thinking=True)

# This captures:
# - All prompts sent to LLMs
# - Complete responses received
# - Token usage and costs
# - Chain processing steps`}
                          </CodeBlock>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-blue-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-start gap-3">
                        <Database className="w-5 h-5 text-blue-400 mt-1" />
                        <div>
                          <h4 className="text-lg font-semibold text-white mb-2">Log File Management</h4>
                          <ul className="text-blue-200 space-y-1 text-sm">
                            <li>• Logs are automatically timestamped and organized</li>
                            <li>• Default location: <code className="text-blue-300">be/logs/</code></li>
                            <li>• Each workflow run creates new log files</li>
                            <li>• JSON files for analysis, JSONL for real-time monitoring</li>
                          </ul>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case "security":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">Security & Privacy</h1>
              <p className="text-lg text-zinc-300 mb-8">
                Understanding NeuroTrace's approach to security and data handling.
              </p>
            </div>

            <div className="space-y-12">
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Data Privacy</h2>
                
                <Card className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border-green-500/30">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-3">
                      <Lock className="w-6 h-6 text-green-400 mt-1" />
                      <div>
                        <h4 className="text-lg font-semibold text-white mb-3">Local Logging Only</h4>
                        <ul className="text-green-200 space-y-2 text-sm">
                          <li>• All logs are stored locally in your <code className="text-green-300">be/logs/</code> directory</li>
                          <li>• No data is sent to external servers or services</li>
                          <li>• Complete control over your workflow data and LLM interactions</li>
                          <li>• Logs remain on your machine unless explicitly shared</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border-blue-500/30">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-3">
                      <Shield className="w-6 h-6 text-blue-400 mt-1" />
                      <div>
                        <h4 className="text-lg font-semibold text-white mb-3">Sensitive Data Handling</h4>
                        <p className="text-blue-200 mb-3">When capturing LLM thinking, be aware that logs may contain:</p>
                        <ul className="text-blue-200 space-y-1 text-sm">
                          <li>• Complete prompts sent to language models</li>
                          <li>• Full responses from LLM services</li>
                          <li>• State data passed between workflow nodes</li>
                          <li>• API keys (if accidentally included in prompts)</li>
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Security Configuration</h2>
                
                <div className="space-y-4">
                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardHeader>
                      <CardTitle>Disable Thinking Capture</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-zinc-300 mb-4">For sensitive workflows, disable LLM capture:</p>
                      <CodeBlock>
{`# Disable LLM thinking capture for sensitive data
neurotrace_logger = NeuroTrace(workflow, capture_thinking=False)

# This will only log:
# - Workflow structure and metadata
# - Node execution timing
# - State transitions (without content)`}
                      </CodeBlock>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardHeader>
                      <CardTitle>Environment Variable Security</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-zinc-300 mb-4">Secure your API keys:</p>
                      <CodeBlock>
{`# Use .env file for API keys (never commit to git)
# .env
OPENAI_API_KEY=your_secret_key_here

# Add .env to your .gitignore
echo ".env" >> .gitignore

# Load in your code
from dotenv import load_dotenv
load_dotenv()`}
                      </CodeBlock>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Security Best Practices</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="bg-gradient-to-br from-amber-500/10 to-orange-500/10 border-amber-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <AlertTriangle className="w-6 h-6 text-amber-400" />
                        <h3 className="text-lg font-semibold text-white">Log Management</h3>
                      </div>
                      <ul className="text-amber-200 space-y-2 text-sm">
                        <li>• Regularly review and clean old log files</li>
                        <li>• Consider log rotation for long-running systems</li>
                        <li>• Secure log directory permissions</li>
                        <li>• Exclude logs from version control</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-red-500/10 to-pink-500/10 border-red-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Lock className="w-6 h-6 text-red-400" />
                        <h3 className="text-lg font-semibold text-white">Data Sanitization</h3>
                      </div>
                      <ul className="text-red-200 space-y-2 text-sm">
                        <li>• Review prompts for accidental key exposure</li>
                        <li>• Filter sensitive data before workflow execution</li>
                        <li>• Use placeholder values for testing</li>
                        <li>• Implement data masking where appropriate</li>
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          </motion.div>
        )

      case "advanced":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">Advanced Usage</h1>
              <p className="text-lg text-zinc-300 mb-8">
                Advanced patterns and techniques for getting the most out of NeuroTrace.
              </p>
            </div>

            <div className="space-y-12">
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Development Workflow</h2>
                
                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>Contributing to NeuroTrace</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p className="text-zinc-300">To contribute to NeuroTrace development:</p>
                    <CodeBlock filename="terminal" language="bash">
{`# 1. Clone the repository
git clone <your-repo-url>
cd neurotrace

# 2. Follow the installation steps
cd be
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Make your changes to the neurotrace/ package
# 4. Test with the provided examples
python examples/mock_example.py
python examples/simple_workflow_example.py

# 5. Add new examples for new features`}
                    </CodeBlock>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Custom Logging Patterns</h2>
                
                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>Conditional Monitoring</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-zinc-300 mb-4">Enable different monitoring levels based on environment:</p>
                    <CodeBlock>
{`import os
from neurotrace import NeuroTrace

# Production: minimal logging
# Development: full monitoring
is_production = os.getenv('ENVIRONMENT') == 'production'

neurotrace_logger = NeuroTrace(
    workflow, 
    capture_thinking=not is_production  # Disable in production
)

# Alternative: Environment-based configuration
capture_llm = os.getenv('NEUROTRACE_CAPTURE_THINKING', 'true').lower() == 'true'
neurotrace_logger = NeuroTrace(workflow, capture_thinking=capture_llm)`}
                    </CodeBlock>
                  </CardContent>
                </Card>

                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>Multi-Workflow Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-zinc-300 mb-4">Monitor multiple workflows in the same session:</p>
                    <CodeBlock>
{`# Create different loggers for different workflows
research_logger = NeuroTrace(research_workflow, capture_thinking=True)
analysis_logger = NeuroTrace(analysis_workflow, capture_thinking=True)

# Run workflows - logs will be separated by timestamp
research_result = research_logger.run(research_input)
analysis_result = analysis_logger.run(analysis_input)

# Each creates separate log files:
# be/logs/node_functions_20240115_103045.json  (research)
# be/logs/workflow_execution_20240115_103045.jsonl
# be/logs/node_functions_20240115_103127.json  (analysis)  
# be/logs/workflow_execution_20240115_103127.jsonl`}
                    </CodeBlock>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Performance Optimization</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Cpu className="w-6 h-6 text-green-400" />
                        <h3 className="text-lg font-semibold text-white">Memory Management</h3>
                      </div>
                      <ul className="text-green-200 space-y-2 text-sm">
                        <li>• NeuroTrace adds minimal overhead to workflows</li>
                        <li>• Log files are written incrementally</li>
                        <li>• State capture is shallow-copied</li>
                        <li>• LLM thinking buffers are managed efficiently</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <Gauge className="w-6 h-6 text-blue-400" />
                        <h3 className="text-lg font-semibold text-white">Performance Tips</h3>
                      </div>
                      <ul className="text-blue-200 space-y-2 text-sm">
                        <li>• Disable thinking capture for high-frequency workflows</li>
                        <li>• Use SSD storage for log directory</li>
                        <li>• Monitor log file sizes in production</li>
                        <li>• Consider log rotation for long-running systems</li>
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Integration Patterns</h2>
                
                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>CI/CD Integration</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-zinc-300 mb-4">Use NeuroTrace in your testing pipeline:</p>
                    <CodeBlock>
{`# test_workflow.py
import pytest
from neurotrace import NeuroTrace
from your_workflow import create_workflow

def test_workflow_execution():
    workflow = create_workflow()
    neurotrace_logger = NeuroTrace(workflow, capture_thinking=False)  # Faster for tests
    
    # Test multiple scenarios
    test_cases = [
        {"input": "test case 1"},
        {"input": "test case 2"}, 
        {"input": "edge case"}
    ]
    
    for case in test_cases:
        result = neurotrace_logger.run(case)
        assert result is not None
        # Logs available for debugging failed tests`}
                    </CodeBlock>
                  </CardContent>
                </Card>

                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>Custom Analysis Tools</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-zinc-300 mb-4">Build custom analysis on top of NeuroTrace logs:</p>
                    <CodeBlock>
{`import json
import glob
from datetime import datetime

def analyze_workflow_performance():
    # Find latest log files
    json_files = glob.glob('be/logs/node_functions_*.json')
    jsonl_files = glob.glob('be/logs/workflow_execution_*.jsonl')
    
    latest_json = max(json_files, key=os.path.getctime)
    latest_jsonl = max(jsonl_files, key=os.path.getctime)
    
    # Load workflow metadata
    with open(latest_json) as f:
        metadata = json.load(f)
    
    # Analyze execution events
    events = []
    with open(latest_jsonl) as f:
        for line in f:
            events.append(json.loads(line))
    
    # Calculate metrics
    start_time = events[0]['timestamp']
    end_time = events[-1]['timestamp']
    duration = datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
    
    print(f"Workflow executed in {duration.total_seconds():.2f} seconds")
    print(f"Total events: {len(events)}")
    
    return {
        'duration': duration.total_seconds(),
        'events': len(events),
        'metadata': metadata
    }`}
                    </CodeBlock>
                  </CardContent>
                </Card>
              </div>
            </div>
          </motion.div>
        )

      case "api":
        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-white mb-4">API Reference</h1>
              <p className="text-lg text-zinc-300 mb-8">
                Complete reference for the NeuroTrace Python library.
              </p>
            </div>

            <div className="space-y-12">
              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">NeuroTrace Class</h2>
                <p className="text-zinc-300">The main entry point for logging and analyzing LangGraph workflows.</p>
                
                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle className="font-mono text-purple-400">NeuroTrace.__init__()</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock>
{`NeuroTrace(
    workflow: StateGraph,
    capture_thinking: bool = True
)`}
                    </CodeBlock>
                    
                    <div className="mt-4">
                      <h4 className="text-lg font-semibold text-white mb-3">Parameters</h4>
                      <div className="space-y-3">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                          <div className="font-mono text-purple-300">workflow</div>
                          <div className="text-zinc-400">StateGraph</div>
                          <div className="text-zinc-300">The LangGraph workflow to instrument and log</div>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                          <div className="font-mono text-purple-300">capture_thinking</div>
                          <div className="text-zinc-400">bool</div>
                          <div className="text-zinc-300">Whether to capture LLM prompts, responses, and token usage (default: True)</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle className="font-mono text-purple-400">NeuroTrace.run()</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock>
{`neurotrace_logger.run(initial_state: Any) -> Any`}
                    </CodeBlock>
                    <p className="text-zinc-300 mt-4">
                      Execute the instrumented workflow with the given initial state. Returns the final state after execution.
                      All workflow execution is logged to files in the logs directory.
                    </p>
                  </CardContent>
                </Card>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">How It Works</h2>
                <p className="text-zinc-300">NeuroTrace analyzes your LangGraph workflow and extracts comprehensive metadata:</p>
                
                <div className="space-y-4">
                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardContent className="p-6">
                      <h4 className="text-lg font-semibold text-white mb-3">Workflow Analysis</h4>
                      <ul className="text-zinc-300 space-y-2 text-sm">
                        <li>• <strong>Node Functions</strong> - Extracts source code and metadata from all workflow nodes</li>
                        <li>• <strong>State Types</strong> - Tracks state type information through workflow execution</li>
                        <li>• <strong>Execution Metadata</strong> - Captures timestamps, processing steps, and workflow structure</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardContent className="p-6">
                      <h4 className="text-lg font-semibold text-white mb-3">LLM Thinking Capture</h4>
                      <p className="text-zinc-300 mb-3">When <code className="text-purple-400">capture_thinking=True</code>, NeuroTrace logs:</p>
                      <ul className="text-zinc-300 space-y-2 text-sm">
                        <li>• <strong>Prompts</strong> - Complete prompts sent to language models</li>
                        <li>• <strong>Responses</strong> - Full responses received from LLMs</li>
                        <li>• <strong>Token Usage</strong> - Input/output tokens and model information</li>
                        <li>• <strong>Chain Processing</strong> - Step-by-step reasoning and intermediate outputs</li>
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Output Files</h2>
                <p className="text-zinc-300">NeuroTrace creates detailed logs in multiple formats stored in <code className="text-purple-400">be/logs/</code>:</p>
                
                <div className="space-y-4">
                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-3">
                        <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">JSON</Badge>
                        <code className="text-purple-300">node_functions_YYYYMMDD_HHMMSS.json</code>
                      </div>
                      <p className="text-zinc-300 mb-3">JSON files for analysis containing:</p>
                      <ul className="text-zinc-400 space-y-1 text-sm">
                        <li>• Source code and metadata of all node functions</li>
                        <li>• State type information and workflow structure</li>
                        <li>• Execution metadata and timestamps</li>
                      </ul>
                    </CardContent>
                  </Card>

                  <Card className="bg-zinc-900/50 border-zinc-700">
                    <CardContent className="p-6">
                      <div className="flex items-center gap-3 mb-3">
                        <Badge className="bg-green-500/20 text-green-300 border-green-500/30">JSONL</Badge>
                        <code className="text-purple-300">workflow_execution_YYYYMMDD_HHMMSS.jsonl</code>
                      </div>
                      <p className="text-zinc-300 mb-3">JSONL files for live monitoring with:</p>
                      <ul className="text-zinc-400 space-y-1 text-sm">
                        <li>• Real-time logging of workflow execution events</li>
                        <li>• Workflow initialization, compilation, execution start/end</li>
                        <li>• LLM thinking capture (prompts, responses, tokens)</li>
                        <li>• Each line is a JSON object with timestamp and event data</li>
                      </ul>
                    </CardContent>
                  </Card>
                </div>
              </div>

              <div className="space-y-6">
                <h2 className="text-2xl font-semibold text-white">Configuration</h2>
                
                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>Environment Variables</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div className="font-mono text-purple-300">OPENAI_API_KEY</div>
                        <div className="text-zinc-400">Optional</div>
                        <div className="text-zinc-300">Required for OpenAI-powered examples and LLM integrations</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-zinc-900/50 border-zinc-700">
                  <CardHeader>
                    <CardTitle>Basic Usage Pattern</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock>
{`from neurotrace import NeuroTrace
from langgraph.graph import StateGraph

# Create your workflow
workflow = StateGraph(YourStateType)
# ... add nodes and edges ...

# Initialize NeuroTrace with LLM thinking capture
neurotrace_logger = NeuroTrace(workflow, capture_thinking=True)

# Run the workflow - logs automatically saved to be/logs/
final_state = neurotrace_logger.run(initial_state)`}
                    </CodeBlock>
                  </CardContent>
                </Card>
              </div>
            </div>
          </motion.div>
        )

      default:
        return (
          <div className="text-center py-12">
            <h2 className="text-xl font-semibold text-white mb-4">Section Coming Soon</h2>
            <p className="text-zinc-400">This documentation section is under development.</p>
          </div>
        )
    }
  }

  return (
    <div className="flex h-screen bg-black relative">
      {/* Background texture */}
      <div className="absolute inset-0 opacity-[0.02] bg-zinc-950 pointer-events-none" 
           style={{
             backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px)',
             backgroundSize: '30px 30px'
           }} />
      
      {/* Docs Navigation Sidebar */}
      <div className="w-80 border-r border-zinc-800 bg-zinc-900/30 p-6 overflow-y-auto relative z-10">
        <DocsNavigation 
          activeSection={activeSection} 
          setActiveSection={setActiveSection} 
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto relative z-10">
        <div className="p-8 max-w-5xl mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeSection}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
} 