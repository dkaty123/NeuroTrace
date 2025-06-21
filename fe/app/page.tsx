"use client"

import Link from "next/link"
import Image from "next/image"
import { useState, useEffect } from "react"
import {
  ArrowRight,
  Github,
  ExternalLink,
  Activity,
  Eye,
  Database,
  CheckCircle,
  Clock,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"


// Helper component for Syntax Highlighting
const PythonCodeBlock = ({ codeString }: { codeString: string }) => {
  const tokenColors = {
    comment: 'text-gray-500',
    string: 'text-emerald-400',
    keyword: 'text-purple-400',
    neurotrace: 'text-blue-400', // For 'NeuroTrace'
    library: 'text-teal-400', // For 'StateGraph', 'LLMChain'
    builtin: 'text-sky-400',  // For 'context', 'invoke', 'run', common builtins
    funcName: 'text-yellow-400',
    decorator: 'text-pink-400',
    number: 'text-orange-400',
    default: 'text-gray-300', // Default text color for unstyled tokens
  };

  const highlightLine = (line: string): string => {
    // Step 1: HTML-escape the entire line to prevent XSS and rendering issues.
    let highlightedLine = line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Step 2: Apply coloring rules. Order can be important.
    // Strings and comments are good to do early.
    highlightedLine = highlightedLine.replace(
      /(".*?"|'.*?')/g,
      (match) => `<span class="${tokenColors.string}">${match}</span>`
    );
    highlightedLine = highlightedLine.replace(
      /(#.*)$/gm,
      (match) => `<span class="${tokenColors.comment}">${match}</span>`
    );

    // Keywords
    const keywordRegex = /\b(from|import|def|return|class|as|with|True|False|None|if|else|elif|for|while|try|except|finally|lambda|yield|pass|break|continue|global|nonlocal|in|is|not|and|or)\b/g;
    highlightedLine = highlightedLine.replace(
      keywordRegex,
      (match) => `<span class="${tokenColors.keyword}">${match}</span>`
    );

    // Specific names (NeuroTrace, library classes, builtins)
    highlightedLine = highlightedLine.replace(
      /\b(NeuroTrace)\b/g,
      (match) => `<span class="${tokenColors.neurotrace}">${match}</span>`
    );
    highlightedLine = highlightedLine.replace(
      /\b(StateGraph|LLMChain)\b/g,
      (match) => `<span class="${tokenColors.library}">${match}</span>`
    );
    highlightedLine = highlightedLine.replace(
      /\b(context|print|len|str|int|float|list|dict|set|tuple|invoke|run)\b/g,
      (match) => `<span class="${tokenColors.builtin}">${match}</span>`
    );

    // Function and class names (after 'def' or 'class')
    // This looks for a span we just created for keywords 'def' or 'class'
    const defClassPattern = `<span class=\"${tokenColors.keyword}\">(def|class)<\/span>\\s+([a-zA-Z_][a-zA-Z0-9_]*)`;
    const defClassRegex = new RegExp(defClassPattern, 'g');
    highlightedLine = highlightedLine.replace(defClassRegex, (_match, keywordSpan, identifier) => {
        return `${keywordSpan} <span class="${tokenColors.funcName}">${identifier}</span>`;
    });
    
    // Decorators
    highlightedLine = highlightedLine.replace(
        /(@[a-zA-Z_]\w*)/g,
        (match) => `<span class="${tokenColors.decorator}">${match}</span>`
    );

    // Numbers
    highlightedLine = highlightedLine.replace(
      /\b(\d+\.?\d*)\b/g,
      (match) => `<span class="${tokenColors.number}">${match}</span>`
    );

    return highlightedLine;
  };

  return (
    <pre className="text-gray-300 whitespace-pre-wrap overflow-x-auto bg-transparent p-0">
      {codeString.split('\n').map((originalLine, i) => (
        <div key={i} dangerouslySetInnerHTML={{ __html: highlightLine(originalLine) }} />
      ))}
    </pre>
  );
};

const integrations = [
  {
    id: "langgraph",
    name: "LangGraph",
    description: "State machine observability with node transitions",
    badge: "Native Support",
    badgeColor: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    code: `from langgraph.graph import StateGraph
from neurotrace import NeuroTrace

# Define your graph
graph = StateGraph()
# ... add nodes and edges ...

# Add NeuroTrace
neurotrace = NeuroTrace(
    graph,
    output_dir="logs",
    capture_thinking=True
)

# Run your workflow
neurotrace.run({"input": "example"})`,
    features: [
      "Real-time state transitions",
      "Node execution timing",
      "Graph visualization",
      "Error propagation tracking",
    ],
  },
  {
    id: "langchain",
    name: "LangChain",
    description: "Chain execution monitoring with tool tracking",
    badge: "Full Integration",
    badgeColor: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    code: `from langchain.chains import LLMChain
from neurotrace import NeuroTrace

# Your LangChain setup
chain = LLMChain(llm=llm, prompt=prompt)

# Add NeuroTrace with callback handler
neurotrace = NeuroTrace(
    chain,
    live_logging=True,
    capture_thinking=True
)

# Run your chain
neurotrace.run("analyze this")`,
    features: ["LLM call monitoring", "Chain step tracking", "Token usage analytics", "Tool invocation logs"],
  },
  {
    id: "custom",
    name: "Custom Python",
    description: "Universal Python agent instrumentation",
    badge: "Universal",
    badgeColor: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    code: `from neurotrace import NeuroTrace

def my_python_agent(query: str) -> str:
    # Your agent logic here
    return f"Result for {query}"

# Add NeuroTrace
neurotrace = NeuroTrace(
    my_python_agent,
    output_dir="logs",
    live_logging=True
)

# Run your agent
result = neurotrace.run("example query")`,
    features: ["Automatic function tracing", "Context propagation", "Custom metrics", "Flexible exporters"],
  },
]

export default function Home() {
  const [activeTab, setActiveTab] = useState(0)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          setActiveTab((current) => (current + 1) % integrations.length)
          return 0
        }
        return prev + 2
      })
    }, 100)

    return () => clearInterval(interval)
  }, [])

  const handleTabClick = (index: number) => {
    setActiveTab(index)
    setProgress(0)
  }

  return (
    <main className="flex min-h-screen flex-col bg-black text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-black/80 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <Link href="/" className="flex items-center space-x-2 group">
                <Image
                  src="/neurotrace-mark.svg"
                  alt="NeuroTrace Logo"
                  width={32}
                  height={32}
                  className="group-hover:scale-110 transition-transform duration-300"
                />
                <span className="text-3xl font-bold text-white group-hover:text-purple-300 transition-colors duration-300">NeuroTrace</span>
              </Link>
            </div>

            <div className="hidden md:flex items-center space-x-8">
              <Link href="#integrations" className="text-sm font-medium text-gray-300 hover:text-white">
                Integrations
              </Link>
              <Link href="#features" className="text-sm font-medium text-gray-300 hover:text-white">
                Features
              </Link>
            </div>

            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" className="text-gray-300 hover:text-white" asChild>
                <Link href="https://github.com/flotoria/neurotrace">
                  <Github className="w-4 h-4 mr-2" />
                  GitHub
                </Link>
              </Button>
              <Button size="sm" className="bg-white text-black hover:bg-gray-100" asChild>
                <Link href="/dashboard">
                  Get Started
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center pt-16">
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-500/20 rounded-full blur-3xl"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-6 lg:px-8 text-center">
          <div className="space-y-8">
 
            {/* Main Headline */}
            <div className="space-y-6">
              <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold">
                <span className="block">Agentic</span>
                <span className="block bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
                  Observability
                </span>
                <span className="block text-4xl md:text-5xl lg:text-6xl text-gray-400 font-normal">Zero Overhead</span>
              </h1>

              <p className="max-w-3xl mx-auto text-xl md:text-2xl text-gray-400 leading-relaxed">
                The first observability framework purpose-built for AI agents.
                <span className="text-white"> Wrap, observe, understand</span> — without changing a single line of your
                agent code.
              </p>
            </div>

            {/* Code Preview */}
            <div className="max-w-4xl mx-auto">
              <div className="relative">
                <div className="bg-gray-900 border border-white/10 rounded-xl overflow-hidden">
                  <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="text-sm text-gray-400 font-mono">agent.py</div>
                  </div>
                  <div className="p-6 font-mono text-left text-sm md:text-base">
                    <div>
                      <span className="text-purple-400">from</span> neurotrace <span className="text-purple-400">import</span> <span className="text-blue-400">NeuroTrace</span>
                    </div>
                    <div className="h-4"></div>
                    <div><span className="text-gray-500"># Your existing workflow</span></div>
                    <div>
                      workflow = <span className="text-blue-400">StateGraph</span>()
                    </div>
                    <div className="h-4"></div>
                    <div><span className="text-gray-500"># Just wrap it with NeuroTrace</span></div>
                    <div>
                      neurotrace = <span className="text-blue-400">NeuroTrace</span>(workflow)
                    </div>
                    <div>
                      neurotrace.<span className="text-sky-400">run</span>()
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
              <Link href="/dashboard">
                <Button size="lg" className="bg-white text-black hover:bg-gray-100">
                  Dashboard
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </Link>
            </div>

           
          </div>
        </div>
      </section>

      {/* Integration Showcase with Auto-cycling Tabs */}
      <section id="integrations" className="py-32 bg-gray-900/50">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              Works with{" "}
              <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Everything
              </span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              From LangGraph state machines to custom Python agents, NeuroTrace adapts to your existing workflow.
            </p>
          </div>

          <div className="max-w-6xl mx-auto">
            {/* Split Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
              {/* Left Side - Loading Indicators */}
              <div className="lg:col-span-4 space-y-4">
                <h3 className="text-2xl font-bold mb-6">Integrations</h3>
                {integrations.map((integration, index) => {
                  const isActive = activeTab === index
                  return (
                    <button
                      key={integration.id}
                      onClick={() => handleTabClick(index)}
                      className={`relative w-full flex items-start space-x-3 px-6 py-4 rounded-xl transition-all duration-300 overflow-hidden text-left ${
                        isActive
                          ? "bg-white/10 text-white border border-white/20"
                          : "text-gray-400 hover:text-white hover:bg-white/5"
                      }`}
                    >
                      <div className="flex-shrink-0 mt-1 text-white">
                        {integration.id === "langgraph" && (
                          <div className="flex items-center space-x-1">
                            <span className="text-2xl">🦜</span>
                            <span className="text-2xl">🕸️</span>
                          </div>
                        )}
                        {integration.id === "langchain" && (
                          <div className="flex items-center space-x-1">
                            <span className="text-2xl">🦜</span>
                            <span className="text-2xl">⛓️</span>
                          </div>
                        )}
                        {integration.id === "custom" && (
                          <Image src="/python-logo.svg" alt="Python Logo" width={20} height={20} className="transition-opacity group-hover:opacity-75" />
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-white">{integration.name}</div>
                        <div className="text-sm text-gray-400">{integration.description}</div>
                      </div>

                      {/* Progress Bar */}
                      {isActive && (
                        <div
                          className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-violet-500 to-indigo-500 transition-all duration-100"
                          style={{ width: `${progress}%` }}
                        />
                      )}
                    </button>
                  )
                })}
              </div>

              {/* Right Side - Code Preview */}
              <div className="lg:col-span-8">
                <div className="bg-gray-900 border border-white/10 rounded-xl overflow-hidden">
                  <div className="flex items-center justify-between px-6 py-4 border-b border-white/10">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <div className="text-sm text-gray-400 font-mono">example.py</div>
                  </div>
                  <div className="p-6 font-mono text-sm">
                    <pre className="text-gray-300">
                      {integrations[activeTab].code}
                    </pre>
                  </div>
                </div>

                {/* Features List */}
                <div className="grid grid-cols-2 gap-4 mt-6">
                  {integrations[activeTab].features.map((feature, index) => (
                    <div key={index} className="flex items-center space-x-3">
                      <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                      <span className="text-gray-300">{feature}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-32">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              Built for the{" "}
              <span className="bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
                Agent Era
              </span>
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Traditional observability tools weren't designed for non-deterministic, multi-step AI workflows. NeuroTrace
              was.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <Card className="bg-gray-900 border-white/10 h-full">
              <CardContent className="p-8">
                <div className="w-12 h-12 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl flex items-center justify-center mb-6">
                  <Eye className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Zero-Code Instrumentation</h3>
                <p className="text-gray-400 mb-6">
                  No decorators, no manual tracing, no code changes. Just wrap your agent and get instant visibility
                  into every execution.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Automatic function wrapping</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Dynamic call graph generation</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Real-time performance metrics</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Feature 2 */}
            <Card className="bg-gray-900 border-white/10 h-full">
              <CardContent className="p-8">
                <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center mb-6">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Agentic Timeline</h3>
                <p className="text-gray-400 mb-6">
                  Visualize the complete thought process of your AI agents. See every decision, tool call, and state
                  transition in chronological order.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Step-by-step execution flow</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Tool usage tracking</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Decision point analysis</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Feature 3 */}
            <Card className="bg-gray-900 border-white/10 h-full">
              <CardContent className="p-8">
                <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-xl flex items-center justify-center mb-6">
                  <Database className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Structured Telemetry</h3>
                <p className="text-gray-400 mb-6">
                  Rich, queryable data that scales from local JSON files to enterprise data pipelines. Built for both
                  debugging and production monitoring.
                </p>
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">JSON-first data format</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Webhook integrations</span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Cloud storage ready</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-16">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-3 mb-6">
                <Link href="/" className="flex items-center space-x-3">
                  <span className="text-white">
                    <Image src="/neurotrace.svg" alt="NeuroTrace Logo" width={32} height={32} />
                  </span>
                  <span className="text-xl font-bold text-white">NeuroTrace</span>
                </Link>
              </div>
              <p className="text-gray-400 mb-6 max-w-md">
                The first observability framework purpose-built for AI agents. Built for the future of autonomous
                systems.
              </p>
              <div className="flex space-x-4">
                <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
                  <Github className="w-5 h-5" />
                </Button>
                <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white">
                  <ExternalLink className="w-5 h-5" />
                </Button>
              </div>
            </div>
          </div> 
        </div>
      </footer>
    </main>
  )
}
