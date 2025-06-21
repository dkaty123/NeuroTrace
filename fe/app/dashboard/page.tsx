"use client";

import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronsLeft, Clock, FileText, Cpu, AlertTriangle, Shield, Code, Eye, EyeOff } from "lucide-react";
import AgentCodeFlow from "../components/AgentCodeFlow";
import Sidebar from "../components/Sidebar";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ProcessedLogEntry {
  event: string;
  timestamp: string;
  overview: string;
  vulnerability: boolean;
  vulnerability_details: string;
  original_data: any;
  processed_at: string;
}

interface SelectedAgentNode {
  agent_id: string;
  agent_name: string;
  source_code: string;
  description?: string;
  file_path?: string;
  dependencies?: string[];
  timestamp: string;
  vulnerabilities?: CodeVulnerability[];
}

interface CodeVulnerability {
  type: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  line?: number;
  suggestion: string;
}

export default function DashboardPage() {
  const [rightTab, setRightTab] = useState<'details' | 'code'>('details');
  const [processedLogs, setProcessedLogs] = useState<ProcessedLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isClient, setIsClient] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<SelectedAgentNode | null>(null);
  const [showingVulnCode, setShowingVulnCode] = useState<string[]>([]);
  const processedLogsRef = useRef<ProcessedLogEntry[]>([]);

  useEffect(() => {
    // Set client flag to prevent hydration mismatch
    setIsClient(true);
  }, []);

  // Update ref whenever processedLogs changes
  useEffect(() => {
    processedLogsRef.current = processedLogs;
  }, [processedLogs]);

  useEffect(() => {
    if (!isClient) return; // Don't fetch on server side

    const fetchProcessedLogs = async (isInitialLoad = false) => {
      try {
        if (isInitialLoad) setLoading(true);
        
        // Fetch the processed logs via API endpoint
        const response = await fetch('/api/get-processed-logs', {
          cache: 'no-store' // Ensure we get the latest version
        });
        
        if (!response.ok) {
          if (response.status === 404) {
            // File doesn't exist yet, that's fine
            if (processedLogsRef.current.length > 0) {
              // Only update if we currently have logs but now there are none
              setProcessedLogs([]);
            }
            setError(null);
            if (isInitialLoad) setLoading(false);
            return;
          }
          throw new Error(`HTTP ${response.status}: Could not load processed logs`);
        }
        
        const data = await response.json();
        if (Array.isArray(data)) {
          const reversedData = data.reverse(); // Show newest first
          
          // Only update if there are actually new logs
          const currentLogs = processedLogsRef.current;
          const hasNewLogs = reversedData.length !== currentLogs.length || 
            (reversedData.length > 0 && currentLogs.length > 0 && 
             reversedData[0].processed_at !== currentLogs[0].processed_at);
          
          if (hasNewLogs || isInitialLoad) {
            setProcessedLogs(reversedData);
            console.log(`Updated logs: ${reversedData.length} entries (${hasNewLogs ? 'new data' : 'initial load'})`);
          }
        } else if (processedLogsRef.current.length > 0) {
          // Only update if we currently have logs
          setProcessedLogs([]);
        }
        setError(null);
      } catch (err) {
        console.error('Error fetching processed logs:', err);
        // Only show error for actual failures, not missing files
        if (err instanceof TypeError && err.message.includes('fetch')) {
          setError(null); // Network error, treat as no logs
          if (processedLogsRef.current.length > 0) setProcessedLogs([]);
        } else {
          setError('Failed to load processed logs');
          if (processedLogsRef.current.length > 0) setProcessedLogs([]);
        }
      } finally {
        if (isInitialLoad) setLoading(false);
      }
    };

    // Initial load
    fetchProcessedLogs(true);

    // Poll for updates every 5 seconds (but only update if there are changes)
    const interval = setInterval(() => fetchProcessedLogs(false), 5000);
    return () => clearInterval(interval);
  }, [isClient]);

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'workflow_initialized':
        return <Cpu className="w-4 h-4" />;
      case 'execution_started':
        return <Clock className="w-4 h-4" />;
      case 'execution_completed':
        return <FileText className="w-4 h-4" />;
      case 'node_function_logged':
        return <Shield className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  const getAlternatingColor = (index: number) => {
    const colors = [
      'border-blue-500 bg-blue-900/20 text-blue-200',
      'border-green-500 bg-green-900/20 text-green-200',
      'border-purple-500 bg-purple-900/20 text-purple-200',
      'border-yellow-500 bg-yellow-900/20 text-yellow-200',
      'border-pink-500 bg-pink-900/20 text-pink-200',
      'border-cyan-500 bg-cyan-900/20 text-cyan-200',
      'border-orange-500 bg-orange-900/20 text-orange-200',
      'border-indigo-500 bg-indigo-900/20 text-indigo-200'
    ];
    return colors[index % colors.length];
  };

  // Safe date formatting to prevent hydration mismatch
  const formatTimestamp = (timestamp: string) => {
    if (!isClient) return timestamp; // Return raw timestamp on server
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp; // Fallback to raw timestamp
    }
  };

  // Analyze source code for vulnerabilities
  const analyzeVulnerabilities = (sourceCode: string): CodeVulnerability[] => {
    const vulnerabilities: CodeVulnerability[] = [];
    const lines = sourceCode.split('\n');

    lines.forEach((line, index) => {
      const lineNumber = index + 1;
      const cleanLine = line.trim().toLowerCase();

      // SQL Injection vulnerabilities
      if (cleanLine.includes('execute(') && (cleanLine.includes('%s') || cleanLine.includes('format(') || cleanLine.includes('f"') || cleanLine.includes("f'"))) {
        vulnerabilities.push({
          type: 'critical',
          title: 'SQL Injection Risk',
          description: 'Direct string concatenation or formatting in SQL queries can lead to SQL injection attacks.',
          line: lineNumber,
          suggestion: 'Use parameterized queries or prepared statements instead of string formatting.'
        });
      }

      // Command Injection
      if (cleanLine.includes('os.system(') || cleanLine.includes('subprocess.call(') || cleanLine.includes('subprocess.run(')) {
        if (cleanLine.includes('input(') || cleanLine.includes('user') || cleanLine.includes('request')) {
          vulnerabilities.push({
            type: 'critical',
            title: 'Command Injection Risk',
            description: 'Executing system commands with user input can lead to command injection attacks.',
            line: lineNumber,
            suggestion: 'Validate and sanitize all user inputs, use subprocess with shell=False, or use allowlists for commands.'
          });
        }
      }

      // Unsafe eval/exec
      if (cleanLine.includes('eval(') || cleanLine.includes('exec(')) {
        vulnerabilities.push({
          type: 'critical',
          title: 'Code Injection Risk',
          description: 'Using eval() or exec() can allow arbitrary code execution.',
          line: lineNumber,
          suggestion: 'Avoid eval() and exec(). Use safer alternatives like ast.literal_eval() for data parsing.'
        });
      }

      // Hardcoded secrets
      const secretPatterns = ['password', 'secret', 'key', 'token', 'api_key'];
      if (secretPatterns.some(pattern => cleanLine.includes(`${pattern} =`) || cleanLine.includes(`"${pattern}"`) || cleanLine.includes(`'${pattern}'`))) {
        if (cleanLine.includes('=') && (cleanLine.includes('"') || cleanLine.includes("'"))) {
          vulnerabilities.push({
            type: 'high',
            title: 'Hardcoded Credentials',
            description: 'Hardcoded passwords, API keys, or secrets in source code pose security risks.',
            line: lineNumber,
            suggestion: 'Use environment variables, configuration files, or secure credential management systems.'
          });
        }
      }

      // Unsafe pickle/deserialization
      if (cleanLine.includes('pickle.load') || cleanLine.includes('pickle.loads')) {
        vulnerabilities.push({
          type: 'high',
          title: 'Unsafe Deserialization',
          description: 'Pickle deserialization of untrusted data can lead to arbitrary code execution.',
          line: lineNumber,
          suggestion: 'Use safer serialization formats like JSON, or validate data sources before unpickling.'
        });
      }

      // Path traversal
      if (cleanLine.includes('open(') && (cleanLine.includes('..') || cleanLine.includes('user') || cleanLine.includes('input'))) {
        vulnerabilities.push({
          type: 'medium',
          title: 'Path Traversal Risk',
          description: 'File operations with user-controlled paths may allow access to unauthorized files.',
          line: lineNumber,
          suggestion: 'Validate file paths, use os.path.abspath() and check if the path is within allowed directories.'
        });
      }

      // Unsafe random for security
      if (cleanLine.includes('random.') && (cleanLine.includes('password') || cleanLine.includes('token') || cleanLine.includes('key'))) {
        vulnerabilities.push({
          type: 'medium',
          title: 'Weak Random Number Generation',
          description: 'Using random module for security-critical operations provides insufficient entropy.',
          line: lineNumber,
          suggestion: 'Use secrets module for cryptographically secure random number generation.'
        });
      }

      // Debug mode indicators
      if (cleanLine.includes('debug=true') || cleanLine.includes('debug = true')) {
        vulnerabilities.push({
          type: 'medium',
          title: 'Debug Mode Enabled',
          description: 'Debug mode can expose sensitive information and stack traces.',
          line: lineNumber,
          suggestion: 'Ensure debug mode is disabled in production environments.'
        });
      }

      // Unsafe imports
      if (cleanLine.includes('import') && cleanLine.includes('*')) {
        vulnerabilities.push({
          type: 'low',
          title: 'Wildcard Import',
          description: 'Wildcard imports can introduce unexpected functions and security risks.',
          line: lineNumber,
          suggestion: 'Import only the specific functions you need or use qualified imports.'
        });
      }

      // HTTP without TLS
      if (cleanLine.includes('http://') && !cleanLine.includes('localhost') && !cleanLine.includes('127.0.0.1')) {
        vulnerabilities.push({
          type: 'medium',
          title: 'Insecure HTTP Connection',
          description: 'HTTP connections transmit data in plain text without encryption.',
          line: lineNumber,
          suggestion: 'Use HTTPS for all external communications to ensure data encryption.'
        });
      }

      // Weak cryptography
      if (cleanLine.includes('md5') || cleanLine.includes('sha1')) {
        vulnerabilities.push({
          type: 'medium',
          title: 'Weak Cryptographic Hash',
          description: 'MD5 and SHA1 are cryptographically weak and vulnerable to collision attacks.',
          line: lineNumber,
          suggestion: 'Use SHA-256 or stronger hashing algorithms for security-critical operations.'
        });
      }
    });

    return vulnerabilities;
  };

  const LogEntrySkeleton = () => (
    <div className="p-3 border-b border-zinc-700/50 animate-pulse bg-zinc-800/30 rounded-md">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-zinc-700 rounded"></div>
          <div className="h-4 w-32 bg-zinc-700 rounded"></div>
        </div>
        <div className="h-3 w-24 bg-zinc-700 rounded"></div>
      </div>
      <div className="h-3 w-full bg-zinc-600 rounded mb-1"></div>
      <div className="h-3 w-3/4 bg-zinc-600 rounded"></div>
    </div>
  );

  const toggleVulnCode = (vulnId: string) => {
    setShowingVulnCode(prev =>
      prev.includes(vulnId)
        ? prev.filter(id => id !== vulnId)
        : [...prev, vulnId]
    );
  };

  // Update the node selection handler
  const handleNodeSelect = (agent: SelectedAgentNode | null) => {
    setSelectedAgent(agent);
    setShowingVulnCode([]); // Reset shown code when a new node is selected
  };

  return (
    <div className="flex min-h-screen bg-background text-foreground dark">
      <Sidebar />
      {/* Main Content Area */}
      <div className="flex flex-1 flex-col min-h-screen bg-black max-h-screen overflow-hidden">
        {/* Main + Right */}
        <div className="flex flex-1 overflow-hidden">
          {/* Agent Code Flow Section */}
          <section className="flex-1 p-6 overflow-hidden bg-black">
            <div className="flex mb-3 gap-2 items-center">
              <h2 className="text-lg font-semibold text-white">Agent Code Flow</h2>
              <div className="text-sm text-gray-400">Live execution pipeline</div>
            </div>
            <Card className="w-full h-[calc(100vh-22rem)] flex bg-black shadow-none border-none rounded-lg overflow-hidden ring-1 ring-purple-500/30">
              <CardContent className="flex-1 p-0 bg-black">
                <AgentCodeFlow onNodeSelect={handleNodeSelect} />
              </CardContent>
            </Card>
          </section>
          {/* Right Sidebar: Details/Code Toggle */}
          <aside className="w-[24rem] border-l border-border flex flex-col p-3 bg-black">
            <div className="flex mb-2 gap-1">
                              <Button
                  variant={rightTab === 'details' ? 'default' : 'outline'}
                  className="flex-1 text-xs py-1.5 px-3 rounded-t-md"
                  onClick={() => setRightTab('details')}
                >
                  Details
                </Button>
                <Button
                  variant={rightTab === 'code' ? 'default' : 'outline'}
                  className="flex-1 text-xs py-1.5 px-3 rounded-t-md"
                  onClick={() => setRightTab('code')}
                >
                  Code
                </Button>
            </div>
            <Card className="flex-1 bg-zinc-900 rounded-b-md overflow-auto scrollbar-hide border border-border/50">
              <CardContent className="!p-0">
                {rightTab === 'details' ? (
                  <div className="w-full p-3">
                    {selectedAgent ? (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2 p-2.5 rounded-md bg-purple-900/40 border border-purple-700/50 text-purple-100">
                          <Code className="w-4 h-4 shrink-0" />
                          <div className="flex flex-col min-w-0">
                            <span className="font-semibold text-sm">{selectedAgent.agent_name}</span>
                            <span className="text-xs opacity-70">ID: {selectedAgent.agent_id}</span>
                          </div>
                        </div>

                        {selectedAgent.description && (
                          <div className="p-2 bg-zinc-800/50 rounded-md border border-zinc-700/50">
                            <h4 className="font-medium text-white text-xs mb-1">Description</h4>
                            <p className="text-xs text-gray-300 leading-relaxed">{selectedAgent.description}</p>
                          </div>
                        )}

                        {/* Security Vulnerabilities Analysis */}
                        {(() => {
                          const vulnerabilities = selectedAgent.vulnerabilities || [];
                          return (
                            <div className="p-2 bg-zinc-800/50 rounded-md border border-zinc-700/50">
                              <div className="flex items-center gap-1.5 mb-2">
                                <Shield className="w-3 h-3 text-red-400" />
                                <h4 className="font-medium text-white text-xs">AI Security Assessment</h4>
                                <span className={`px-1.5 py-0.5 text-xs rounded-full ${
                                  vulnerabilities.length === 0 
                                    ? 'bg-green-900/50 text-green-300 border border-green-700/50'
                                    : vulnerabilities.some(v => v.type === 'critical')
                                    ? 'bg-red-900/50 text-red-300 border border-red-700/50'
                                    : vulnerabilities.some(v => v.type === 'high')
                                    ? 'bg-orange-900/50 text-orange-300 border border-orange-700/50'
                                    : 'bg-yellow-900/50 text-yellow-300 border border-yellow-700/50'
                                }`}>
                                  {vulnerabilities.length === 0 ? 'Secure' : `${vulnerabilities.length} Issue${vulnerabilities.length > 1 ? 's' : ''}`}
                                </span>
                      </div>

                              {vulnerabilities.length === 0 ? (
                                <div className="flex items-center gap-1.5 text-green-300 text-xs">
                                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full"></div>
                                  No vulnerabilities detected.
                                </div>
                              ) : (
                                <div className="space-y-2">
                                  {vulnerabilities.map((vuln, index) => {
                                    const vulnId = `${selectedAgent.agent_id}-${index}`;
                                    const isShowingCode = showingVulnCode.includes(vulnId);

                                    return (
                                      <div key={vulnId} className={`p-3 rounded-lg border ${
                                        vuln.type === 'critical' 
                                          ? 'bg-red-900/20 border-red-700' 
                                          : vuln.type === 'high'
                                          ? 'bg-orange-900/20 border-orange-700'
                                          : vuln.type === 'medium'
                                          ? 'bg-yellow-900/20 border-yellow-700'
                                          : 'bg-blue-900/20 border-blue-700'
                                      }`}>
                                        <div className="flex items-start gap-2 mb-2">
                                          <span className={`px-2 py-1 text-xs font-medium rounded uppercase ${
                                            vuln.type === 'critical' 
                                              ? 'bg-red-500 text-white' 
                                              : vuln.type === 'high'
                                              ? 'bg-orange-500 text-white'
                                              : vuln.type === 'medium'
                                              ? 'bg-yellow-500 text-black'
                                              : 'bg-blue-500 text-white'
                                          }`}>
                                            {vuln.type}
                                          </span>
                                          <div className="flex-1">
                                            <h5 className="font-medium text-white text-sm">{vuln.title}</h5>
                                            {vuln.line && (
                                              <span className="text-xs text-gray-400">Line {vuln.line}</span>
                                            )}
                                          </div>
                                        </div>
                                        <p className="text-xs text-gray-300 mb-2">{vuln.description}</p>
                                        <div className="text-xs text-gray-400">
                                          <span className="font-medium">Recommendation:</span> {vuln.suggestion}
                                        </div>

                                        {/* Code Section */}
                                        {vuln.line && (
                                          <div className="mt-3">
                                            <Button
                                              variant="ghost"
                                              size="sm"
                                              onClick={() => toggleVulnCode(vulnId)}
                                              className="h-7 px-2 text-xs hover:bg-red-500/10 text-red-400"
                                            >
                                              {isShowingCode ? (
                                                <>
                                                  <EyeOff className="w-3.5 h-3.5 mr-1.5" />
                                                  Hide Code
                                                </>
                                              ) : (
                                                <>
                                                  <Eye className="w-3.5 h-3.5 mr-1.5" />
                                                  Show Code
                                                </>
                                              )}
                                            </Button>
                                            
                                            {isShowingCode && (
                                              <div className="mt-2 font-mono text-sm">
                                                <div className="flex items-center text-xs text-zinc-500 mb-1 px-2">
                                                  <span>Context around line {vuln.line}</span>
                                                </div>
                                                <div className="rounded bg-black/50 border border-red-500/20">
                                                  {(() => {
                                                    const lines = selectedAgent.source_code.split('\n');
                                                    const start = Math.max(0, vuln.line - 3);
                                                    const end = Math.min(lines.length, vuln.line + 2);
                                                    
                                                    return lines.slice(start, end).map((line, index) => {
                                                      const currentLineNumber = start + index + 1;
                                                      const isVulnerableLine = currentLineNumber === vuln.line;
                                                      
                                                      return (
                                                        <div 
                                                          key={currentLineNumber}
                                                          className={`flex ${isVulnerableLine ? 'bg-red-500/10 border-l-2 border-red-500' : 'border-l-2 border-transparent'}`}
                                                        >
                                                          <div className="px-2 py-1 text-xs text-zinc-500 select-none w-12 text-right border-r border-zinc-700/50">
                                                            {currentLineNumber}
                                                          </div>
                                                          <pre className={`px-3 py-1 whitespace-pre-wrap flex-1 min-w-0 break-words ${isVulnerableLine ? 'text-red-200' : 'text-zinc-300'}`}>
                                                            {line}
                                                          </pre>
                                                        </div>
                                                      );
                                                    });
                                                  })()}
                                                </div>
                                              </div>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              )}
                            </div>
                          );
                        })()}

                        <div className="p-2 bg-zinc-800/50 rounded-md border border-zinc-700/50">
                          <h4 className="font-medium text-white text-xs mb-1">Metadata</h4>
                          <div className="space-y-1 text-xs">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Lines of Code:</span>
                              <span className="text-white">{selectedAgent.source_code.split('\n').length}</span>
                            </div>
                            {selectedAgent.file_path && (
                              <div className="flex justify-between">
                                <span className="text-gray-400">File Path:</span>
                                <span className="text-white font-mono text-xs">{selectedAgent.file_path}</span>
                              </div>
                            )}
                            <div className="flex justify-between">
                              <span className="text-gray-400">Timestamp:</span>
                              <span className="text-white text-xs">{formatTimestamp(selectedAgent.timestamp)}</span>
                            </div>
                          </div>
                        </div>

                        {selectedAgent.dependencies && selectedAgent.dependencies.length > 0 && (
                          <div className="p-2 bg-zinc-800/50 rounded-md border border-zinc-700/50">
                            <h4 className="font-medium text-white text-xs mb-1">Dependencies</h4>
                            <div className="flex flex-wrap gap-1">
                              {selectedAgent.dependencies.map((dep, index) => (
                                <span key={index} className="px-2 py-1 bg-blue-900/50 text-blue-200 text-xs rounded border border-blue-700">
                                  {dep}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full text-gray-400">
                        <div className="text-center">
                          <Code className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p className="text-sm">Click on a node to view details</p>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-3 h-full overflow-auto">
                    {selectedAgent ? (
                      <div className="max-w-none prose prose-invert prose-sm">
                        {/* Agent Header */}
                        <div className="not-prose mb-4">
                          <div className="flex items-center gap-2 mb-2">
                            <div className="p-1.5 bg-purple-500/20 rounded-md">
                              <Code className="w-3 h-3 text-purple-400" />
                            </div>
                            <div>
                              <h1 className="text-sm font-bold text-white m-0">{selectedAgent.agent_name}</h1>
                              <p className="text-xs text-gray-400 m-0">{selectedAgent.file_path || 'source.py'}</p>
                        </div>
                      </div>

                          {selectedAgent.description && (
                            <div className="bg-zinc-800/50 border border-zinc-700/50 rounded-md p-2 mb-3">
                              <h3 className="text-xs font-medium text-gray-300 mb-1">Description</h3>
                              <p className="text-xs text-gray-400 leading-relaxed">{selectedAgent.description}</p>
                            </div>
                          )}

                          <div className="flex flex-wrap gap-3 text-xs text-gray-400 mb-4">
                            <span className="flex items-center gap-1">
                              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                              {selectedAgent.source_code.split('\n').length} lines
                            </span>
                            {selectedAgent.dependencies && selectedAgent.dependencies.length > 0 && (
                              <span className="flex items-center gap-1">
                                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                {selectedAgent.dependencies.length} dependencies
                              </span>
                            )}
                            <span className="flex items-center gap-1">
                              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                              {formatTimestamp(selectedAgent.timestamp)}
                            </span>
                          </div>
                        </div>

                        {/* Code Section */}
                        <div className="not-prose">
                          <h2 className="text-sm font-semibold text-white mb-2 flex items-center gap-1">
                            <span className="text-purple-400">#</span> Source Code
                          </h2>
                          
                          <div className="bg-zinc-900/50 rounded-md border border-zinc-700/50 overflow-hidden">
                            <SyntaxHighlighter
                              language="python"
                              style={vscDarkPlus}
                              customStyle={{
                                margin: 0,
                                padding: '0.75rem',
                                background: 'transparent',
                                fontSize: '11px',
                                lineHeight: '1.4',
                                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
                              }}
                              wrapLines={true}
                              wrapLongLines={true}
                              showLineNumbers={false}
                              codeTagProps={{
                                style: {
                                  wordBreak: 'break-word',
                                  whiteSpace: 'pre-wrap',
                                  overflowWrap: 'break-word',
                                }
                              }}
                            >
                              {selectedAgent.source_code}
                            </SyntaxHighlighter>
                      </div>

                          {/* Dependencies Section */}
                          {selectedAgent.dependencies && selectedAgent.dependencies.length > 0 && (
                            <div className="mt-8">
                              <h3 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
                                <span className="text-purple-400">#</span> Dependencies
                              </h3>
                              <div className="bg-zinc-900/30 rounded-lg border border-zinc-700 p-4">
                                <div className="flex flex-wrap gap-2">
                                  {selectedAgent.dependencies.map((dep, index) => (
                                    <code key={index} className="px-3 py-1.5 bg-blue-500/20 text-blue-300 text-sm rounded-md border border-blue-500/30 font-mono">
                                      {dep}
                                    </code>
                                  ))}
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Function Analysis */}
                          <div className="mt-8">
                            <h3 className="text-md font-semibold text-white mb-3 flex items-center gap-2">
                              <span className="text-purple-400">#</span> Analysis
                            </h3>
                            <div className="bg-zinc-900/30 rounded-lg border border-zinc-700 p-4">
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-gray-400">Function Type:</span>
                                  <p className="text-white font-mono">
                                    {selectedAgent.source_code.includes('def ') ? 'Function Definition' : 'Script'}
                                  </p>
                                </div>
                                <div>
                                  <span className="text-gray-400">Complexity:</span>
                                  <p className="text-white">
                                    {selectedAgent.source_code.split('\n').length < 10 ? 'Simple' : 
                                     selectedAgent.source_code.split('\n').length < 30 ? 'Medium' : 'Complex'}
                                  </p>
                                </div>
                              </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                      <div className="flex items-center justify-center h-full text-gray-400">
                        <div className="text-center">
                          <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                          <h3 className="text-lg font-medium mb-2">No Agent Selected</h3>
                          <p className="text-sm">Click on a node in the flow diagram to view its source code and details</p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </aside>
        </div>
        {/* Output Panel at Bottom */}
        <footer className="w-full border-t border-border p-3 bg-black max-h-64 overflow-hidden">
          <Card className="w-full bg-zinc-900 border border-border h-full">
                          <CardContent className="py-2 px-3 h-full flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-md font-semibold text-white">Processed Logs</h3>
                  <div className="text-xs text-gray-400">
                    {(!isClient || (loading && processedLogs.length === 0)) ? '' : processedLogs.length === 0 && !error ? '' : `${processedLogs.length} entries`}
                  </div>
                </div>
              <div className="flex-1 overflow-y-auto scrollbar-hide min-h-[160px]">
                {(!isClient || (loading && processedLogs.length === 0)) ? (
                  <div className="space-y-2 p-1">
                    {[...Array(3)].map((_, i) => <LogEntrySkeleton key={i} />)}
                  </div>
                ) : error ? (
                  <div className="p-1 h-full flex items-center justify-center">
                    <div className="p-6 text-center bg-red-900/20 text-red-300 rounded-lg w-full max-w-md mx-auto">
                      <AlertTriangle className="w-10 h-10 mx-auto mb-3 opacity-75" />
                      <h3 className="text-md font-medium text-red-200 mb-1">Error Loading Logs</h3>
                      <p className="text-xs">{error}</p>
                    </div>
                  </div>
                ) : processedLogs.length === 0 ? (
                  <div className="space-y-2 p-1 flex items-center justify-center h-full">
                    <div className="p-8 text-center text-zinc-400 flex flex-col items-center justify-center min-h-[200px] w-full max-w-sm">
                      <FileText className="w-12 h-12 mx-auto mb-4 text-zinc-500 opacity-60" />
                      <h3 className="text-md font-medium text-zinc-200 mb-1.5">No Processed Logs</h3>
                      <p className="text-xs text-zinc-500">
                        Run a workflow to see results and analysis here.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col gap-2 p-1">
                    {processedLogs.slice(0, 20).map((log, index) => (
                      <Card 
                        key={index} 
                        className={`border ${getAlternatingColor(index)} transition-all hover:scale-[1.01] cursor-pointer`}
                      >
                        <CardContent className="p-3">
                          <div className="flex items-start gap-3">
                            <div className="mt-0.5 flex items-center gap-1">
                              {getEventIcon(log.event)}
                              {log.vulnerability && (
                                <div className="relative">
                                  <AlertTriangle className="w-3 h-3 text-red-500 animate-pulse" />
                                  <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-red-500 rounded-full"></div>
                                </div>
                              )}
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-start justify-between mb-1">
                                <div className="flex items-center gap-2">
                                  <h4 className="font-semibold text-xs">
                                    {log.event.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                  </h4>
                                  {log.vulnerability && (
                                    <span className="text-xs bg-red-500/20 text-red-300 px-1.5 py-0.5 rounded-full border border-red-500/30 font-medium">
                                      VULNERABLE
                                    </span>
                                  )}
                                </div>
                                <p className="text-xs opacity-70 ml-2 shrink-0">
                                  {formatTimestamp(log.timestamp)}
                                </p>
                              </div>
                              <p className="text-sm leading-relaxed mb-2">
                                {log.overview}
                              </p>
                              {log.vulnerability && (
                                <div className="mb-1 p-1.5 bg-red-500/10 border border-red-500/30 rounded text-xs">
                                  <div className="flex items-center gap-1 mb-0.5">
                                    <AlertTriangle className="w-2.5 h-2.5 text-red-400" />
                                    <span className="font-medium text-red-300 text-xs">Alert:</span>
                                  </div>
                                  <p className="text-red-200 text-xs leading-tight">
                                    {log.vulnerability_details}
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </footer>
      </div>
    </div>
  );
}
