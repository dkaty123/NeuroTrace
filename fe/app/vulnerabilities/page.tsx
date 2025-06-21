"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Shield,
  AlertTriangle,
  Clock,
  Terminal,
  Search,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Eye,
  Target,
  Bug,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Loader2,
  Code,
  FileText,
  Activity,
  RefreshCw,
  Filter,
  Settings,
  X,
  Radar,
  Zap,
  TrendingUp,
  BarChart3
} from "lucide-react";
import Sidebar from "../components/Sidebar";
import { motion, AnimatePresence } from "framer-motion";

type Severity = "critical" | "high" | "medium" | "low";
type VulnerabilitySource = "agent_code" | "workflow_logs" | "runtime_error";

interface CodeVulnerability {
  type: Severity;
  title: string;
  description: string;
  line?: number;
  suggestion: string;
}

interface ProcessedLog {
  event: string;
  timestamp: string;
  overview: string;
  vulnerability: boolean;
  vulnerability_details: string;
  original_data: any;
  processed_at: string;
}

interface ProcessedAgent {
  agent_id: string;
  agent_name: string;
  source_code: string;
  description?: string;
  file_path?: string;
  dependencies?: string[];
  timestamp: string;
  processed_at: string;
  vulnerabilities?: CodeVulnerability[];
}

interface UnifiedVulnerability {
  id: string;
  source: VulnerabilitySource;
  severity: Severity;
  title: string;
  description: string;
  timestamp: string;
  node_name: string;
  file_path?: string;
  line_number?: number;
  suggestion: string;
  code_snippet?: string;
  raw_data: ProcessedAgent | ProcessedLog;
}

// Animated Counter Component
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

// Filter Chip Component
const FilterChip = ({ label, onRemove, icon, color = "purple" }: { 
  label: string; 
  onRemove: () => void; 
  icon?: string;
  color?: string;
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.8 }}
    animate={{ opacity: 1, scale: 1 }}
    exit={{ opacity: 0, scale: 0.8 }}
    className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium
      ${color === 'red' ? 'bg-red-900/30 text-red-300 border border-red-700/50' :
        color === 'orange' ? 'bg-orange-900/30 text-orange-300 border border-orange-700/50' :
        color === 'blue' ? 'bg-blue-900/30 text-blue-300 border border-blue-700/50' :
        'bg-purple-900/30 text-purple-300 border border-purple-700/50'
      }`}
  >
    {icon && <span>{icon}</span>}
    {label}
    <button onClick={onRemove} className="hover:bg-white/10 rounded-full p-0.5">
      <X className="w-3 h-3" />
    </button>
  </motion.div>
);

// Threat Radar Component
const ThreatRadar = ({ newThreats, isActive }: { newThreats: number; isActive: boolean }) => (
  <motion.div 
    className="fixed bottom-6 right-6 z-50"
    initial={{ scale: 0, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    transition={{ delay: 0.5 }}
  >
    <div className="relative">
      <motion.div
        className="w-16 h-16 rounded-full bg-gradient-to-br from-red-900/80 to-orange-900/80 
                   border-2 border-red-500/50 flex items-center justify-center cursor-pointer
                   shadow-lg backdrop-blur-sm"
        whileHover={{ scale: 1.1 }}
        animate={isActive ? {
          boxShadow: [
            "0 0 0 0 rgba(239, 68, 68, 0.7)",
            "0 0 0 10px rgba(239, 68, 68, 0)",
          ]
        } : {}}
        transition={{
          duration: 1.5,
          repeat: isActive ? Infinity : 0,
          repeatType: "loop"
        }}
      >
        <Radar className="w-6 h-6 text-red-300" />
      </motion.div>
      
      {/* Radar waves */}
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-red-500/30"
        animate={isActive ? {
          scale: [1, 1.5, 2],
          opacity: [0.8, 0.4, 0]
        } : {}}
        transition={{
          duration: 2,
          repeat: isActive ? Infinity : 0,
          repeatType: "loop"
        }}
      />
      
      {newThreats > 0 && (
        <motion.div
          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 rounded-full 
                     flex items-center justify-center text-xs font-bold text-white"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          {newThreats}
        </motion.div>
      )}
    </div>
  </motion.div>
);

const getSeverityColor = (severity: Severity) => {
  const colors = {
    critical: "bg-red-500/10 text-red-400 border-red-500/20",
    high: "bg-orange-500/10 text-orange-400 border-orange-500/20", 
    medium: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    low: "bg-blue-500/10 text-blue-400 border-blue-500/20"
  };
  return colors[severity];
};

const getSeverityIcon = (severity: Severity) => {
  const icons = {
    critical: "🔴",
    high: "🟠", 
    medium: "🟡",
    low: "🔵"
  };
  return icons[severity];
};

const getSourceIcon = (source: VulnerabilitySource) => {
  const icons = {
    agent_code: <Code className="w-4 h-4" />,
    workflow_logs: <Activity className="w-4 h-4" />,
    runtime_error: <AlertTriangle className="w-4 h-4" />
  };
  return icons[source];
};

export default function VulnerabilitiesPage() {
  const [severityFilter, setSeverityFilter] = useState<"all" | Severity>("all");
  const [sourceFilter, setSourceFilter] = useState<"all" | VulnerabilitySource>("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [processedLogs, setProcessedLogs] = useState<ProcessedLog[]>([]);
  const [processedAgents, setProcessedAgents] = useState<ProcessedAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedCards, setExpandedCards] = useState<string[]>([]);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isResetting, setIsResetting] = useState(false);
  const [newDataReceived, setNewDataReceived] = useState(false);
  const [visibleCards, setVisibleCards] = useState<string[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  const currentLogsRef = useRef<ProcessedLog[]>([]);
  const currentAgentsRef = useRef<ProcessedAgent[]>([]);

  // Fetch data from APIs
  useEffect(() => {
    let isInitialLoad = true;

    const fetchData = async () => {
      try {
        if (isInitialLoad) setLoading(true);
        
        const [logsResponse, agentsResponse] = await Promise.all([
          fetch('/api/get-processed-logs', { cache: 'no-store' }),
          fetch('/api/get-agent-code', { cache: 'no-store' })
        ]);

        if (!logsResponse.ok || !agentsResponse.ok) {
          throw new Error('Failed to fetch data');
        }

        const logsData = await logsResponse.json();
        const agentsData = await agentsResponse.json();

        const logsChanged = JSON.stringify(logsData) !== JSON.stringify(currentLogsRef.current);
        const agentsChanged = JSON.stringify(agentsData) !== JSON.stringify(currentAgentsRef.current);

        if (logsChanged || agentsChanged || isInitialLoad) {
          setProcessedLogs(logsData);
          setProcessedAgents(agentsData);
          currentLogsRef.current = logsData;
          currentAgentsRef.current = agentsData;
          setLastUpdated(new Date());
          
          // Trigger new data animation
          if (!isInitialLoad && (logsChanged || agentsChanged)) {
            setNewDataReceived(true);
            setTimeout(() => setNewDataReceived(false), 3000);
          }
        }
      } catch (error) {
        console.error('Error fetching vulnerability data:', error);
      } finally {
        if (isInitialLoad) {
          setLoading(false);
          isInitialLoad = false;
        }
      }
    };

    fetchData();
    if (autoRefresh) {
      const interval = setInterval(fetchData, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Transform data into unified vulnerability format
  const vulnerabilities = useMemo(() => {
    const unified: UnifiedVulnerability[] = [];

    // Process agent code vulnerabilities  
    processedAgents.forEach(agent => {
      if (agent.vulnerabilities && agent.vulnerabilities.length > 0) {
        agent.vulnerabilities.forEach((vuln, index) => {
          const codeLines = agent.source_code.split('\n');
          const lineNumber = vuln.line || 1;
          const codeSnippet = codeLines.slice(
            Math.max(0, lineNumber - 3),
            Math.min(codeLines.length, lineNumber + 2)
          ).join('\n');

          unified.push({
            id: `agent-${agent.agent_id}-${index}`,
            source: "agent_code",
            severity: vuln.type,
            title: vuln.title,
            description: vuln.description,
            timestamp: agent.timestamp,
            node_name: agent.agent_name,
            file_path: agent.file_path,
            line_number: vuln.line,
            suggestion: vuln.suggestion,
            code_snippet: codeSnippet,
            raw_data: agent
          });
        });
      }
    });

    // Process log vulnerabilities
    processedLogs.forEach((log, index) => {
      if (log.vulnerability && log.vulnerability_details !== "No vulnerabilities detected") {
        // Determine severity based on log content
        let severity: Severity = "medium";
        const details = log.vulnerability_details.toLowerCase();
        if (details.includes("critical") || details.includes("remote code") || details.includes("injection")) {
          severity = "critical";
        } else if (details.includes("high") || details.includes("security") || details.includes("unauthorized")) {
          severity = "high";
        } else if (details.includes("low") || details.includes("info")) {
          severity = "low";
        }

        unified.push({
          id: `log-${index}`,
          source: "workflow_logs",
          severity,
          title: `Security Alert: ${log.event}`,
          description: log.vulnerability_details,
          timestamp: log.timestamp,
          node_name: log.event,
          suggestion: "Review workflow security and implement proper validation",
          raw_data: log
        });
      }
    });

    return unified.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [processedAgents, processedLogs]);

  // Filter vulnerabilities
  const filteredVulnerabilities = useMemo(() => {
    return vulnerabilities.filter(vuln => {
      const matchesSeverity = severityFilter === "all" || vuln.severity === severityFilter;
      const matchesSource = sourceFilter === "all" || vuln.source === sourceFilter;
      const matchesSearch = searchTerm === "" || 
        vuln.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        vuln.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        vuln.node_name.toLowerCase().includes(searchTerm.toLowerCase());
      
      return matchesSeverity && matchesSource && matchesSearch;
    });
  }, [vulnerabilities, severityFilter, sourceFilter, searchTerm]);

  // Staggered card appearance animation
  useEffect(() => {
    const cardIds = filteredVulnerabilities.map(v => v.id);
    setVisibleCards([]);
    
    cardIds.forEach((id, index) => {
      setTimeout(() => {
        setVisibleCards(prev => [...prev, id]);
      }, index * 100); // 100ms stagger between each card
    });
  }, [filteredVulnerabilities]);

  const toggleExpanded = (id: string) => {
    setExpandedCards(prev => 
      prev.includes(id) 
        ? prev.filter(cardId => cardId !== id)
        : [...prev, id]
    );
  };

  const stats = useMemo(() => {
    const total = vulnerabilities.length;
    const critical = vulnerabilities.filter(v => v.severity === "critical").length;
    const high = vulnerabilities.filter(v => v.severity === "high").length;
    const medium = vulnerabilities.filter(v => v.severity === "medium").length;
    const low = vulnerabilities.filter(v => v.severity === "low").length;
    
    return { total, critical, high, medium, low };
  }, [vulnerabilities]);

  const activeFilters = useMemo(() => {
    const filters = [];
    if (severityFilter !== "all") {
      filters.push({
        type: "severity",
        label: severityFilter.charAt(0).toUpperCase() + severityFilter.slice(1),
        icon: getSeverityIcon(severityFilter),
        color: severityFilter === "critical" ? "red" : severityFilter === "high" ? "orange" : severityFilter === "medium" ? "yellow" : "blue",
        onRemove: () => setSeverityFilter("all")
      });
    }
    if (sourceFilter !== "all") {
      filters.push({
        type: "source",
        label: sourceFilter === "agent_code" ? "Agent Code" : sourceFilter === "workflow_logs" ? "Workflow Logs" : "Runtime Error",
        icon: sourceFilter === "agent_code" ? "💻" : sourceFilter === "workflow_logs" ? "📜" : "⚠️",
        color: "blue",
        onRemove: () => setSourceFilter("all")
      });
    }
    return filters;
  }, [severityFilter, sourceFilter]);

  const handleReset = async () => {
    if (!confirm('Are you sure you want to reset all vulnerability data? This will clear all current data and prepare for new agent runs.')) {
      return;
    }

    setIsResetting(true);
    try {
      const response = await fetch('/api/reset-data', {
        method: 'POST',
      });

      if (response.ok) {
        // Clear local state immediately
        setProcessedLogs([]);
        setProcessedAgents([]);
        setExpandedCards([]);
        setVisibleCards([]);
        currentLogsRef.current = [];
        currentAgentsRef.current = [];
        setLastUpdated(new Date());
        console.log('✅ Data reset successfully - ready for new agents');
      } else {
        console.error('Failed to reset data');
      }
    } catch (error) {
      console.error('Error resetting data:', error);
    } finally {
      setIsResetting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen bg-black">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-8 h-8 animate-spin text-purple-400 mx-auto mb-4" />
            <p className="text-zinc-400">Initializing threat analysis...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-black relative">
      {/* Background texture */}
      <div className="absolute inset-0 opacity-[0.02] bg-zinc-950" 
           style={{
             backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(255,255,255,0.01) 2px, rgba(255,255,255,0.01) 4px)',
             backgroundSize: '30px 30px'
           }} />
      
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header with KPI Bar */}
        <div className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-xl">
          {/* Top Header */}
          <div className="p-6 pb-4">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-gradient-to-br from-red-500/20 to-orange-500/20 text-red-400">
                  <Shield className="w-6 h-6" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-white">Threat Console</h1>
                  <p className="text-zinc-400">
                    Real-time security monitoring & vulnerability analysis
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <motion.button
                  onClick={() => setAutoRefresh(!autoRefresh)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all ${
                    autoRefresh 
                      ? 'bg-green-900/30 text-green-400 border border-green-700/50' 
                      : 'bg-zinc-800 text-zinc-400 border border-zinc-700'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <motion.div
                    animate={autoRefresh ? { rotate: 360 } : {}}
                    transition={{ duration: 2, repeat: autoRefresh ? Infinity : 0, ease: "linear" }}
                  >
                    <RefreshCw className="w-4 h-4" />
                  </motion.div>
                  {autoRefresh ? 'Live' : 'Paused'}
                </motion.button>
                
                <span className="text-xs text-zinc-500">
                  {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Never'}
                </span>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  disabled={isResetting}
                  className="text-zinc-400 hover:text-red-400 border-zinc-700 hover:border-red-700"
                >
                  {isResetting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Resetting...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Reset
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* KPI Bar */}
          <div className="px-6 pb-4">
            <motion.div 
              className="relative bg-gradient-to-r from-zinc-900/80 via-zinc-800/80 to-zinc-900/80 
                         rounded-xl border border-zinc-700/50 p-1 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="grid grid-cols-5 divide-x divide-zinc-700/50">
                {/* Total */}
                <div className="px-6 py-4 text-center">
                  <div className="text-2xl font-bold text-white mb-1">
                    <AnimatedCounter end={stats.total} />
                  </div>
                  <div className="text-xs text-zinc-400 uppercase tracking-wide">Total</div>
                </div>
                
                {/* Critical */}
                <div className="px-6 py-4 text-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-red-900/0 via-red-900/20 to-red-900/0" />
                  <div className="relative">
                    <div className="text-2xl font-bold text-red-400 mb-1">
                      <AnimatedCounter end={stats.critical} />
                    </div>
                    <div className="text-xs text-red-300 uppercase tracking-wide">Critical</div>
                  </div>
                </div>
                
                {/* High */}
                <div className="px-6 py-4 text-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-orange-900/0 via-orange-900/20 to-orange-900/0" />
                  <div className="relative">
                    <div className="text-2xl font-bold text-orange-400 mb-1">
                      <AnimatedCounter end={stats.high} />
                    </div>
                    <div className="text-xs text-orange-300 uppercase tracking-wide">High</div>
                  </div>
                </div>
                
                {/* Medium */}
                <div className="px-6 py-4 text-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-yellow-900/0 via-yellow-900/20 to-yellow-900/0" />
                  <div className="relative">
                    <div className="text-2xl font-bold text-yellow-400 mb-1">
                      <AnimatedCounter end={stats.medium} />
                    </div>
                    <div className="text-xs text-yellow-300 uppercase tracking-wide">Medium</div>
                  </div>
                </div>
                
                {/* Low */}
                <div className="px-6 py-4 text-center relative overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-900/0 via-blue-900/20 to-blue-900/0" />
                  <div className="relative">
                    <div className="text-2xl font-bold text-blue-400 mb-1">
                      <AnimatedCounter end={stats.low} />
                    </div>
                    <div className="text-xs text-blue-300 uppercase tracking-wide">Low</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Search and Filter Bar */}
          <div className="px-6 pb-6">
            <div className="flex items-center gap-4">
              {/* Search */}
              <div className="relative flex-1">
                <Search className="absolute left-3 top-3 w-4 h-4 text-zinc-400" />
                <Input
                  placeholder="🔍 Search vulnerabilities..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-zinc-800/50 border-zinc-700 text-white placeholder:text-zinc-500 
                           focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                />
              </div>
              
              {/* Filter Toggle */}
              <motion.button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-all ${
                  showFilters || activeFilters.length > 0
                    ? 'bg-purple-900/30 text-purple-300 border border-purple-700/50' 
                    : 'bg-zinc-800 text-zinc-400 border border-zinc-700'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Filter className="w-4 h-4" />
                Filters
                {activeFilters.length > 0 && (
                  <span className="bg-purple-500 text-white text-xs rounded-full px-1.5 py-0.5 min-w-[1rem] h-4 flex items-center justify-center">
                    {activeFilters.length}
                  </span>
                )}
              </motion.button>
              
              <span className="text-xs text-zinc-500">
                Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Never'}
              </span>
            </div>
            
            {/* Active Filter Chips */}
            {activeFilters.length > 0 && (
              <motion.div 
                className="flex flex-wrap gap-2 mt-3"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
              >
                <AnimatePresence>
                  {activeFilters.map((filter, index) => (
                    <FilterChip
                      key={filter.type}
                      label={filter.label}
                      icon={filter.icon}
                      color={filter.color}
                      onRemove={filter.onRemove}
                    />
                  ))}
                </AnimatePresence>
              </motion.div>
            )}
          </div>
        </div>

        {/* Filter Drawer */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="border-b border-zinc-800 bg-zinc-900/30 backdrop-blur-sm overflow-hidden"
            >
              <div className="p-6 space-y-4">
                <div className="grid grid-cols-2 gap-6">
                  {/* Severity Filter */}
                  <div>
                    <label className="block text-sm font-medium text-zinc-300 mb-2">Severity Level</label>
                    <select
                      value={severityFilter}
                      onChange={(e) => setSeverityFilter(e.target.value as "all" | Severity)}
                      className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm 
                               focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                    >
                      <option value="all">All Severities</option>
                      <option value="critical">🔴 Critical</option>
                      <option value="high">🟠 High</option>
                      <option value="medium">🟡 Medium</option>
                      <option value="low">🔵 Low</option>
                    </select>
                  </div>
                  
                  {/* Source Filter */}
                  <div>
                    <label className="block text-sm font-medium text-zinc-300 mb-2">Source Type</label>
                    <select
                      value={sourceFilter}
                      onChange={(e) => setSourceFilter(e.target.value as "all" | VulnerabilitySource)}
                      className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm 
                               focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                    >
                      <option value="all">All Sources</option>
                      <option value="agent_code">💻 Agent Code</option>
                      <option value="workflow_logs">📜 Workflow Logs</option>
                      <option value="runtime_error">⚠️ Runtime Errors</option>
                    </select>
                  </div>
                </div>
                
                <div className="flex justify-end">
                  <button
                    onClick={() => {
                      setSeverityFilter("all");
                      setSourceFilter("all");
                      setSearchTerm("");
                    }}
                    className="text-sm text-zinc-400 hover:text-white transition-colors"
                  >
                    Clear all filters
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Content - Vulnerability List */}
        <div className="flex-1 overflow-auto p-6 hide-scrollbar">
          {newDataReceived && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-4 p-3 bg-gradient-to-r from-purple-900/20 to-blue-900/20 border border-purple-700/50 
                         rounded-lg flex items-center gap-2 backdrop-blur-sm"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Radar className="w-4 h-4 text-purple-400" />
              </motion.div>
              <span className="text-sm text-purple-300">Detecting new threats...</span>
            </motion.div>
          )}
          
          {filteredVulnerabilities.length === 0 ? (
            <div className="text-center py-12">
              {vulnerabilities.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 
                                  flex items-center justify-center">
                    <Shield className="w-8 h-8 text-purple-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">Threat Analysis Standby</h3>
                  <p className="text-zinc-400 mb-6">
                    No vulnerabilities detected. System ready for agent execution.
                  </p>
                  <div className="text-sm text-zinc-500 bg-zinc-800/50 rounded-lg p-4 max-w-md mx-auto">
                    <p className="mb-2 text-zinc-300">🎯 <strong>Next Steps:</strong></p>
                    <ul className="text-left space-y-1">
                      <li>• Execute Python agents with NeuroTrace integration</li>
                      <li>• Real-time code analysis & vulnerability detection</li>
                      <li>• Security alerts & remediation suggestions</li>
                      <li>• Live threat monitoring dashboard</li>
                    </ul>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <Shield className="w-16 h-16 mx-auto text-green-400 mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">No Threats Found</h3>
                  <p className="text-zinc-400">
                    {searchTerm || severityFilter !== "all" || sourceFilter !== "all"
                      ? "Try adjusting your filters to see more results."
                      : "All systems secure. No vulnerabilities detected!"
                    }
                  </p>
                </motion.div>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <AnimatePresence>
                {filteredVulnerabilities.map((vuln, index) => {
                  const isExpanded = expandedCards.includes(vuln.id);
                  const isVisible = visibleCards.includes(vuln.id);
                  
                  if (!isVisible) return null;
                  
                  return (
                    <motion.div
                      key={vuln.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ 
                        opacity: 1, 
                        x: 0,
                        transition: {
                          duration: 0.3,
                          delay: index * 0.05,
                        }
                      }}
                      exit={{ 
                        opacity: 0, 
                        x: 20,
                        transition: { duration: 0.2 }
                      }}
                      className="group"
                    >
                      <motion.div
                        className={`relative bg-zinc-900/60 border rounded-lg cursor-pointer overflow-hidden
                          backdrop-blur-sm transition-all duration-300 hover:bg-zinc-900/80
                          ${vuln.severity === 'critical' ? 'border-red-700/50 hover:border-red-600 hover:shadow-lg hover:shadow-red-500/20' :
                            vuln.severity === 'high' ? 'border-orange-700/50 hover:border-orange-600 hover:shadow-lg hover:shadow-orange-500/20' :
                            vuln.severity === 'medium' ? 'border-yellow-700/50 hover:border-yellow-600 hover:shadow-lg hover:shadow-yellow-500/20' :
                            'border-blue-700/50 hover:border-blue-600 hover:shadow-lg hover:shadow-blue-500/20'
                          }`}
                        whileHover={{ y: -1 }}
                        transition={{ duration: 0.2 }}
                      >
                        {/* Heat Strip */}
                        <div className={`absolute left-0 top-0 bottom-0 w-1 
                          ${vuln.severity === 'critical' ? 'bg-gradient-to-b from-red-500 to-red-700' :
                            vuln.severity === 'high' ? 'bg-gradient-to-b from-orange-500 to-orange-700' :
                            vuln.severity === 'medium' ? 'bg-gradient-to-b from-yellow-500 to-yellow-700' :
                            'bg-gradient-to-b from-blue-500 to-blue-700'
                          }`} 
                        />
                        
                        {/* Main Content */}
                        <div 
                          onClick={() => toggleExpanded(vuln.id)}
                          className="pl-6 pr-4 py-4 flex items-center justify-between hover:bg-zinc-800/30 transition-colors"
                        >
                          <div className="flex items-center gap-4 flex-1 min-w-0">
                            {/* Severity Icon */}
                            <div className="flex items-center gap-2">
                              <span className="text-lg">{getSeverityIcon(vuln.severity)}</span>
                              <div className="text-zinc-400">
                                {getSourceIcon(vuln.source)}
                              </div>
                            </div>
                            
                            {/* Content */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-3 mb-1">
                                <h3 className="font-medium text-white group-hover:text-purple-400 transition-colors truncate">
                                  {vuln.title}
                                </h3>
                                <Badge className={`${getSeverityColor(vuln.severity)} text-xs px-2 py-0.5`}>
                                  {vuln.severity}
                                </Badge>
                                <Badge variant="outline" className="text-xs px-2 py-0.5 border-zinc-600 text-zinc-400">
                                  {vuln.source === 'agent_code' ? '💻 Code' : '📜 Workflow'}
                                </Badge>
                              </div>
                              
                              <div className="flex items-center gap-4 text-xs text-zinc-500">
                                <span className="flex items-center gap-1">
                                  <Target className="w-3 h-3" />
                                  {vuln.node_name}
                                </span>
                                <span className="flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {new Date(vuln.timestamp).toLocaleString()}
                                </span>
                                {vuln.line_number && (
                                  <span className="flex items-center gap-1">
                                    <Code className="w-3 h-3" />
                                    Line {vuln.line_number}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          
                          {/* Expand Icon */}
                          <motion.div
                            animate={{ rotate: isExpanded ? 180 : 0 }}
                            transition={{ duration: 0.2 }}
                            className="text-zinc-400 group-hover:text-zinc-300"
                          >
                            <ChevronDown className="w-5 h-5" />
                          </motion.div>
                        </div>
                        
                        {/* Expanded Content */}
                        <AnimatePresence>
                          {isExpanded && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.3 }}
                              className="border-t border-zinc-700/50"
                            >
                              <div className="px-6 py-4 space-y-4">
                                {/* Description */}
                                <div>
                                  <h4 className="text-sm font-medium text-zinc-300 mb-2">Description</h4>
                                  <p className="text-sm text-zinc-400">{vuln.description}</p>
                                </div>
                                
                                {/* Recommendation - Sticky on scroll */}
                                <Alert className="bg-blue-900/20 border-blue-700/50 sticky top-0 z-10 backdrop-blur-sm">
                                  <AlertCircle className="w-4 h-4" />
                                  <AlertDescription className="text-blue-300">
                                    <strong>Recommendation:</strong> {vuln.suggestion}
                                  </AlertDescription>
                                </Alert>
                                
                                {/* Code Snippet with Tabs */}
                                {vuln.code_snippet && (
                                  <div>
                                    <h4 className="text-sm font-medium text-zinc-300 mb-3 flex items-center gap-2">
                                      <Terminal className="w-4 h-4" />
                                      Code Context
                                    </h4>
                                    <div className="bg-black/50 border border-zinc-700 rounded-lg overflow-hidden">
                                      <div className="border-b border-zinc-700/50 px-3 py-2 bg-zinc-800/50">
                                        <div className="flex items-center gap-2">
                                          <div className="w-3 h-3 rounded-full bg-red-500/60"></div>
                                          <div className="w-3 h-3 rounded-full bg-yellow-500/60"></div>
                                          <div className="w-3 h-3 rounded-full bg-green-500/60"></div>
                                          <span className="text-xs text-zinc-400 ml-2">
                                            {vuln.file_path || 'snippet.py'}
                                          </span>
                                        </div>
                                      </div>
                                      <pre className="p-4 text-sm text-zinc-300 overflow-x-auto">
                                        {vuln.code_snippet}
                                      </pre>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}
        </div>
      </div>
      
      {/* Threat Radar Widget */}
      <ThreatRadar 
        newThreats={newDataReceived ? vulnerabilities.length : 0}
        isActive={newDataReceived}
      />
    </div>
  );
} 