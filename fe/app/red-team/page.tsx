"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Shield,
  AlertTriangle,
  Target,
  Activity,
  Clock,
  Zap,
  CheckCircle2,
  XCircle,
  Loader2,
  Play,
  Pause,
  RefreshCw,
  Filter,
  Search,
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  EyeOff,
  Settings,
  Download,
  BarChart3,
  LineChart,
  AlertCircle
} from "lucide-react";
import Sidebar from "../components/Sidebar";
import { motion, AnimatePresence } from "framer-motion";
import { RedTeamResult, RedTeamSuite, RED_TEAM_SUITES, RED_TEAM_PROMPTS } from "../lib/types/redTeam";
import { RedTeamService } from "../lib/services/redTeamService";

interface RedTeamStats {
  totalTests: number;
  passed: number;
  failed: number;
  riskScore: number;
  lastRun: Date | null;
  trending: 'up' | 'down' | 'stable';
}

const AnimatedCounter = ({ end, duration = 1000, className = "" }: { end: number; duration?: number; className?: string }) => {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    let startTime = Date.now();
    const startCount = 0;
    
    const updateCount = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      
      setCount(Math.floor(startCount + (end - startCount) * easeOutQuart));
      
      if (progress < 1) {
        requestAnimationFrame(updateCount);
      }
    };
    
    updateCount();
  }, [end, duration]);
  
  return <span className={className}>{count}</span>;
};

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case 'critical': return 'bg-red-500/10 text-red-400 border-red-500/20';
    case 'high': return 'bg-orange-500/10 text-orange-400 border-orange-500/20';
    case 'medium': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
    case 'low': return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
    default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
  }
};

const getCategoryIcon = (category: string) => {
  switch (category) {
    case 'jailbreak': return '🔓';
    case 'prompt_injection': return '💉';
    case 'data_extraction': return '📤';
    case 'privilege_escalation': return '⬆️';
    case 'adversarial': return '⚔️';
    case 'safety': return '🛡️';
    case 'bias': return '⚖️';
    case 'misuse': return '🚫';
    default: return '🎯';
  }
};

export default function RedTeamPage() {
  const [selectedSuite, setSelectedSuite] = useState<RedTeamSuite>(RED_TEAM_SUITES[0]);
  const [isRunning, setIsRunning] = useState(false);
  const [autoMode, setAutoMode] = useState(false);
  const [results, setResults] = useState<RedTeamResult[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [showDetails, setShowDetails] = useState<string[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [totalTests, setTotalTests] = useState(0);

  const redTeamService = RedTeamService.getInstance();

  // Calculate stats
  const stats: RedTeamStats = useMemo(() => {
    const total = results.length;
    const passed = results.filter(r => r.passed).length;
    const failed = total - passed;
    const riskScore = total > 0 ? Math.round((failed / total) * 100) : 0;
    
    return {
      totalTests: total,
      passed,
      failed,
      riskScore,
      lastRun: results.length > 0 ? new Date(Math.max(...results.map(r => new Date(r.timestamp).getTime()))) : null,
      trending: riskScore > 50 ? 'up' : riskScore < 20 ? 'down' : 'stable'
    };
  }, [results]);

  // Filter results
  const filteredResults = useMemo(() => {
    return results.filter(result => {
      const prompt = RED_TEAM_PROMPTS.find(p => p.id === result.prompt_id);
      if (!prompt) return false;

      const matchesSearch = searchTerm === "" || 
        prompt.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        prompt.technique.toLowerCase().includes(searchTerm.toLowerCase()) ||
        result.input.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesSeverity = severityFilter === "all" || prompt.severity === severityFilter;
      const matchesCategory = categoryFilter === "all" || prompt.category === categoryFilter;
      
      return matchesSearch && matchesSeverity && matchesCategory;
    });
  }, [results, searchTerm, severityFilter, categoryFilter]);

  const runTestSuite = async (suite: RedTeamSuite) => {
    setIsRunning(true);
    setCurrentProgress(0);
    setTotalTests(suite.prompts.length);
    
    try {
      const newResults: RedTeamResult[] = [];
      
      for (let i = 0; i < suite.prompts.length; i++) {
        const prompt = suite.prompts[i];
        const result = await redTeamService.runTest(prompt.id);
        newResults.push(result);
        setCurrentProgress(((i + 1) / suite.prompts.length) * 100);
        
        // Small delay for better UX
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      setResults(prev => [...newResults, ...prev].slice(0, 100)); // Keep last 100 results
    } catch (error) {
      console.error('Error running test suite:', error);
    } finally {
      setIsRunning(false);
      setCurrentProgress(0);
    }
  };

  const toggleDetails = (resultId: string) => {
    setShowDetails(prev => 
      prev.includes(resultId)
        ? prev.filter(id => id !== resultId)
        : [...prev, resultId]
    );
  };

  const exportResults = () => {
    const dataStr = JSON.stringify(filteredResults, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `red-team-results-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  useEffect(() => {
    // Auto-refresh for automated suites
    if (autoMode && selectedSuite.automated) {
      const interval = setInterval(() => {
        if (!isRunning) {
          runTestSuite(selectedSuite);
        }
      }, 60000); // Run every minute in demo
      
      return () => clearInterval(interval);
    }
  }, [autoMode, selectedSuite, isRunning]);

  return (
    <div className="flex h-screen bg-black">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex-shrink-0 p-8 pb-4 border-b border-zinc-800/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-red-500/20 to-purple-500/20">
                <Shield className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Red Team Console</h1>
                <p className="text-zinc-400">Adversarial AI Security Testing</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAutoMode(!autoMode)}
                className={`${autoMode ? 'bg-green-500/20 text-green-400 border-green-500' : 'text-zinc-400'}`}
              >
                {autoMode ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                {autoMode ? 'Auto Mode' : 'Manual Mode'}
              </Button>
              <Button
                onClick={exportResults}
                disabled={filteredResults.length === 0}
                className="text-zinc-400 hover:text-purple-400"
                variant="outline"
                size="sm"
              >
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </div>

        {/* Stats Dashboard */}
        <div className="flex-shrink-0 p-8 pt-4">
          <div className="grid grid-cols-5 gap-4">
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-400">Total Tests</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">
                  <AnimatedCounter end={stats.totalTests} />
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-400">Passed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <div className="text-2xl font-bold text-green-400">
                    <AnimatedCounter end={stats.passed} />
                  </div>
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-400">Failed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <div className="text-2xl font-bold text-red-400">
                    <AnimatedCounter end={stats.failed} />
                  </div>
                  <XCircle className="w-4 h-4 text-red-400" />
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-400">Risk Score</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2">
                  <div className={`text-2xl font-bold ${stats.riskScore > 50 ? 'text-red-400' : stats.riskScore > 25 ? 'text-yellow-400' : 'text-green-400'}`}>
                    <AnimatedCounter end={stats.riskScore} />%
                  </div>
                  {stats.trending === 'up' ? (
                    <TrendingUp className="w-4 h-4 text-red-400" />
                  ) : stats.trending === 'down' ? (
                    <TrendingDown className="w-4 h-4 text-green-400" />
                  ) : (
                    <Minus className="w-4 h-4 text-zinc-400" />
                  )}
                </div>
              </CardContent>
            </Card>
            
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-zinc-400">Last Run</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm text-zinc-300">
                  {stats.lastRun ? stats.lastRun.toLocaleTimeString() : 'Never'}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-8 pt-0 space-y-6">
          {/* Test Suite Selection & Controls */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card className="bg-zinc-900/60 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Target className="w-5 h-5 text-purple-400" />
                    Test Suites
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  {RED_TEAM_SUITES.map((suite) => (
                    <motion.div
                      key={suite.id}
                      className={`p-4 rounded-lg border cursor-pointer transition-all duration-200 ${
                        selectedSuite.id === suite.id
                          ? 'bg-purple-500/20 border-purple-500/50'
                          : 'bg-zinc-800/30 border-zinc-700/50 hover:border-zinc-600/50'
                      }`}
                      onClick={() => setSelectedSuite(suite)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium text-white">{suite.name}</h3>
                        <div className="flex items-center gap-2">
                          {suite.automated && (
                            <Badge className="bg-green-500/20 text-green-400">Auto</Badge>
                          )}
                          <Badge variant="outline" className="text-zinc-400">
                            {suite.prompts.length} tests
                          </Badge>
                        </div>
                      </div>
                      <p className="text-sm text-zinc-400">{suite.description}</p>
                    </motion.div>
                  ))}
                </CardContent>
              </Card>
            </div>
            
            <div>
              <Card className="bg-zinc-900/60 border-zinc-800">
                <CardHeader>
                  <CardTitle className="text-white">Controls</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Button
                    onClick={() => runTestSuite(selectedSuite)}
                    disabled={isRunning}
                    className="w-full bg-red-600 hover:bg-red-700 text-white"
                  >
                    {isRunning ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Running...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Run Suite
                      </>
                    )}
                  </Button>
                  
                  {isRunning && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm text-zinc-400">
                        <span>Progress</span>
                        <span>{Math.round(currentProgress)}%</span>
                      </div>
                      <div className="w-full bg-zinc-800 rounded-full h-2">
                        <div 
                          className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${currentProgress}%` }}
                        />
                      </div>
                    </div>
                  )}
                  
                  <div className="text-xs text-zinc-500 space-y-1">
                    <div>Suite: {selectedSuite.name}</div>
                    <div>Tests: {selectedSuite.prompts.length}</div>
                    <div>Schedule: {selectedSuite.schedule || 'manual'}</div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Filters */}
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardContent className="p-4">
              <div className="flex items-center gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-3 w-4 h-4 text-zinc-400" />
                  <Input
                    placeholder="Search tests..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-zinc-800/50 border-zinc-700"
                  />
                </div>
                
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  className="bg-zinc-800 border-zinc-700 text-white rounded px-3 py-2"
                >
                  <option value="all">All Severities</option>
                  <option value="critical">Critical</option>
                  <option value="high">High</option>
                  <option value="medium">Medium</option>
                  <option value="low">Low</option>
                </select>
                
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  className="bg-zinc-800 border-zinc-700 text-white rounded px-3 py-2"
                >
                  <option value="all">All Categories</option>
                  <option value="jailbreak">Jailbreak</option>
                  <option value="prompt_injection">Prompt Injection</option>
                  <option value="data_extraction">Data Extraction</option>
                  <option value="privilege_escalation">Privilege Escalation</option>
                  <option value="adversarial">Adversarial</option>
                  <option value="safety">Safety</option>
                  <option value="bias">Bias</option>
                  <option value="misuse">Misuse</option>
                </select>
              </div>
            </CardContent>
          </Card>

          {/* Results */}
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Activity className="w-5 h-5 text-purple-400" />
                Test Results ({filteredResults.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              {filteredResults.length === 0 ? (
                <div className="text-center py-12">
                  <Shield className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
                  <p className="text-zinc-500">No test results available</p>
                  <p className="text-zinc-600 text-sm mt-1">Run a test suite to see results here</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <AnimatePresence>
                    {filteredResults.map((result) => {
                      const prompt = RED_TEAM_PROMPTS.find(p => p.id === result.prompt_id);
                      if (!prompt) return null;
                      
                      const showDetail = showDetails.includes(result.id);
                      
                      return (
                        <motion.div
                          key={result.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -20 }}
                          className="border border-zinc-700/50 rounded-lg overflow-hidden"
                        >
                          <div
                            className="p-4 cursor-pointer hover:bg-zinc-800/30 transition-colors"
                            onClick={() => toggleDetails(result.id)}
                          >
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-3">
                                <span className="text-xl">{getCategoryIcon(prompt.category)}</span>
                                <div>
                                  <h4 className="font-medium text-white">{prompt.description}</h4>
                                  <p className="text-sm text-zinc-400">{prompt.technique}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Badge className={getSeverityColor(prompt.severity)}>
                                  {prompt.severity}
                                </Badge>
                                <Badge className={result.passed ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}>
                                  {result.passed ? 'Passed' : 'Failed'}
                                </Badge>
                                {showDetail ? <EyeOff className="w-4 h-4 text-zinc-400" /> : <Eye className="w-4 h-4 text-zinc-400" />}
                              </div>
                            </div>
                            
                            <div className="flex items-center gap-4 text-xs text-zinc-500">
                              <span>Risk: {result.risk_score || 0}/100</span>
                              <span>Confidence: {result.confidence || 0}%</span>
                              <span>{new Date(result.timestamp).toLocaleString()}</span>
                            </div>
                          </div>
                          
                          <AnimatePresence>
                            {showDetail && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="border-t border-zinc-700/50 bg-zinc-800/20 p-4"
                              >
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  <div>
                                    <h5 className="text-sm font-medium text-zinc-300 mb-2">Input</h5>
                                    <pre className="text-xs text-zinc-400 bg-black/30 p-3 rounded overflow-x-auto">
                                      {result.input}
                                    </pre>
                                  </div>
                                  <div>
                                    <h5 className="text-sm font-medium text-zinc-300 mb-2">Output</h5>
                                    <pre className="text-xs text-zinc-400 bg-black/30 p-3 rounded overflow-x-auto">
                                      {result.output}
                                    </pre>
                                  </div>
                                </div>
                                
                                {prompt.tags.length > 0 && (
                                  <div className="mt-3">
                                    <h5 className="text-sm font-medium text-zinc-300 mb-2">Tags</h5>
                                    <div className="flex flex-wrap gap-1">
                                      {prompt.tags.map(tag => (
                                        <Badge key={tag} variant="outline" className="text-xs">
                                          {tag}
                                        </Badge>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
} 