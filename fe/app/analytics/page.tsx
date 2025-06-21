"use client";

import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Activity,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Brain,
  Clock,
  Cpu,
  FileText,
  LineChart,
  Shield,
  Terminal,
  Users,
  Zap,
  RefreshCw,
  Loader2
} from "lucide-react";
import Sidebar from "../components/Sidebar";
import { motion } from "framer-motion";

interface AnalyticsData {
  overview: {
    totalRequests: number;
    activeAgents: number;
    avgResponseTime: number;
    uptime: number;
    errorRate: number;
  };
  performance: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    qps: number;
  };
  agentMetrics: {
    completionRate: number;
    avgTokensPerRequest: number;
    toolUsage: Record<string, number>;
    topErrors: Array<{ type: string; count: number }>;
  };
  securityMetrics: {
    vulnerabilitiesDetected: number;
    criticalCount: number;
    highCount: number;
    mediumCount: number;
    lowCount: number;
    avgResponseTime: number;
    mitigationRate: number;
  };
  trends: {
    daily: {
      requests: number[];
      errors: number[];
      latency: number[];
    };
  };
}

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

const MetricCard = ({ title, value, icon: Icon, trend, color = "purple" }: {
  title: string;
  value: string | number;
  icon: any;
  trend?: { value: number; isPositive: boolean };
  color?: string;
}) => (
  <Card className="bg-zinc-900/60 border-zinc-800 backdrop-blur-xl">
    <CardHeader className="flex flex-row items-center justify-between pb-2">
      <CardTitle className="text-sm font-medium text-zinc-400">{title}</CardTitle>
      <Icon className={`h-4 w-4 text-${color}-400`} />
    </CardHeader>
    <CardContent>
      <div className="flex items-baseline justify-between">
        <div className="text-2xl font-bold text-white">{value}</div>
        {trend && (
          <Badge 
            className={trend.isPositive ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"}
          >
            {trend.isPositive ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
            {Math.abs(trend.value)}%
          </Badge>
        )}
      </div>
    </CardContent>
  </Card>
);



export default function AnalyticsPage() {
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [newDataReceived, setNewDataReceived] = useState(false);
  const [isManualRefresh, setIsManualRefresh] = useState(false);
  const [recentActivity, setRecentActivity] = useState<any[]>([]);
  
  // Use ref to track current data for comparison without causing re-renders
  const currentDataRef = useRef<AnalyticsData | null>(null);

  useEffect(() => {
    let isInitialLoad = true;

    const fetchAnalytics = async () => {
      try {
        if (isInitialLoad) setLoading(true);
        
        const [analyticsResponse, logsResponse, agentsResponse] = await Promise.all([
          fetch('/api/get-analytics', { 
            cache: 'no-store',
            headers: {
              'Cache-Control': 'no-cache'
            }
          }),
          fetch('/api/get-processed-logs', { cache: 'no-store' }),
          fetch('/api/get-agent-code', { cache: 'no-store' })
        ]);

        if (!analyticsResponse.ok) {
          throw new Error('Failed to fetch analytics data');
        }

        const data = await analyticsResponse.json();
        const logs = logsResponse.ok ? await logsResponse.json() : [];
        const agents = agentsResponse.ok ? await agentsResponse.json() : [];
        
        // Calculate unified vulnerabilities using same logic as vulnerabilities page
        const unifiedVulnerabilities: any[] = [];

        // Process agent code vulnerabilities  
        agents.forEach((agent: any) => {
          if (agent.vulnerabilities && agent.vulnerabilities.length > 0) {
            agent.vulnerabilities.forEach((vuln: any) => {
              unifiedVulnerabilities.push({
                id: `agent-${agent.agent_id}-${vuln.title}`,
                source: "agent_code",
                severity: vuln.type,
                title: vuln.title,
                description: vuln.description,
                timestamp: agent.timestamp,
                agent_name: agent.agent_name,
                suggestion: vuln.suggestion
              });
            });
          }
        });

        // Process log vulnerabilities
        logs.forEach((log: any, index: number) => {
          if (log.vulnerability && log.vulnerability_details !== "No vulnerabilities detected") {
            // Determine severity based on log content
            let severity = "medium";
            const details = log.vulnerability_details.toLowerCase();
            if (details.includes("critical") || details.includes("remote code") || details.includes("injection")) {
              severity = "critical";
            } else if (details.includes("high") || details.includes("security") || details.includes("unauthorized")) {
              severity = "high";
            } else if (details.includes("low") || details.includes("info")) {
              severity = "low";
            }

            unifiedVulnerabilities.push({
              id: `log-${index}`,
              source: "workflow_logs",
              severity,
              title: `Security Alert: ${log.event}`,
              description: log.vulnerability_details,
              timestamp: log.timestamp,
              agent_name: log.event
            });
          }
        });

        // Calculate enhanced metrics
        const totalExecutions = logs.length + agents.length;
        const successfulExecutions = logs.filter((log: any) => !log.vulnerability).length + 
                                   agents.filter((agent: any) => (agent.vulnerabilities?.length || 0) === 0).length;
        const completionRate = totalExecutions > 0 ? Math.round((successfulExecutions / totalExecutions) * 100) : 100;
        
        // Calculate vulnerability counts from unified vulnerabilities
        const totalVulnerabilities = unifiedVulnerabilities.length;
        const criticalVulns = unifiedVulnerabilities.filter((v: any) => v.severity === 'critical').length;
        const highVulns = unifiedVulnerabilities.filter((v: any) => v.severity === 'high').length;
        const mediumVulns = unifiedVulnerabilities.filter((v: any) => v.severity === 'medium').length;
        const lowVulns = unifiedVulnerabilities.filter((v: any) => v.severity === 'low').length;
        
        // Create recent activity timeline
        const activities = [
          ...logs.map((log: any) => ({
            type: 'log',
            timestamp: log.timestamp,
            title: log.event,
            description: log.overview,
            hasVulnerability: log.vulnerability,
            data: log
          })),
          ...agents.map((agent: any) => ({
            type: 'agent',
            timestamp: agent.timestamp,
            title: `Agent: ${agent.agent_name}`,
            description: agent.description || 'Agent code processed',
            hasVulnerability: (agent.vulnerabilities?.length || 0) > 0,
            data: agent
          }))
        ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
         .slice(0, 10); // Show last 10 activities

        setRecentActivity(activities);
        
        // Update data with calculated metrics
        const enhancedData = {
          ...data,
          agentMetrics: {
            ...data.agentMetrics,
            completionRate: completionRate
          },
          securityMetrics: {
            ...data.securityMetrics,
            vulnerabilitiesDetected: totalVulnerabilities,
            criticalCount: criticalVulns,
            highCount: highVulns,
            mediumCount: mediumVulns,
            lowCount: lowVulns
          }
        };
        
        // Check if data actually changed using ref comparison
        const dataChanged = JSON.stringify(enhancedData) !== JSON.stringify(currentDataRef.current);
        
        if (dataChanged || isInitialLoad) {
          currentDataRef.current = enhancedData;
          setAnalyticsData(enhancedData);
          setLastUpdated(new Date());
          
          // Trigger new data animation only on actual changes
          if (!isInitialLoad && dataChanged) {
            setNewDataReceived(true);
            setTimeout(() => setNewDataReceived(false), 2000);
          }
        }
      } catch (error) {
        console.error('Error fetching analytics data:', error);
      } finally {
        if (isInitialLoad) {
          setLoading(false);
          isInitialLoad = false;
        }
        setIsManualRefresh(false);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []); // Remove analyticsData dependency to prevent interval recreation

  const handleManualRefresh = () => {
    setIsManualRefresh(true);
    // The useEffect will handle the actual refresh
  };

  if (loading) {
    return (
      <div className="flex h-screen bg-black">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="w-8 h-8 animate-spin text-purple-400 mx-auto mb-4" />
            <p className="text-zinc-400">Loading analytics data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!analyticsData) {
    return (
      <div className="flex h-screen bg-black">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <AlertTriangle className="w-16 h-16 text-amber-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-white mb-2">Unable to Load Analytics</h3>
            <p className="text-zinc-400 mb-4">Failed to fetch analytics data</p>
            <Button onClick={handleManualRefresh} disabled={isManualRefresh}>
              {isManualRefresh ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <RefreshCw className="w-4 h-4 mr-2" />}
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-black">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Fixed Header */}
        <div className="flex-shrink-0 p-8 pb-4 border-b border-zinc-800/50">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Analytics</h1>
              <p className="text-zinc-400">System health and performance overview</p>
            </div>
            <div className="flex items-center gap-3">
              <motion.div 
                className="flex items-center gap-2 text-sm text-zinc-400"
                animate={newDataReceived ? {
                  scale: [1, 1.05, 1],
                  color: ["#a1a1aa", "#10b981", "#a1a1aa"]
                } : {}}
                transition={{ duration: 0.6 }}
              >
                <RefreshCw className={`w-4 h-4 ${newDataReceived ? 'animate-spin' : ''}`} />
                <span>
                  Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Never'}
                </span>
              </motion.div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleManualRefresh}
                disabled={isManualRefresh}
                className="text-zinc-400 hover:text-purple-400 border-zinc-700 hover:border-purple-700"
              >
                {isManualRefresh ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Refreshing...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-8 pt-6 space-y-6">
            {/* Key Metrics - Scrollable */}
            <motion.div 
              className="grid grid-cols-4 gap-4"
              initial="hidden"
              animate="visible"
              variants={{
                visible: {
                  transition: {
                    staggerChildren: 0.1
                  }
                }
              }}
              key={analyticsData.overview.totalRequests}
            >
              <motion.div variants={fadeIn}>
                <MetricCard
                  title="Total Requests"
                  value={analyticsData.overview.totalRequests.toLocaleString()}
                  icon={Activity}
                  trend={analyticsData.overview.totalRequests > 0 ? { value: 12.5, isPositive: true } : undefined}
                />
              </motion.div>
              <motion.div variants={fadeIn}>
                <MetricCard
                  title="Active Agents"
                  value={analyticsData.overview.activeAgents}
                  icon={Users}
                  color="blue"
                />
              </motion.div>
              <motion.div variants={fadeIn}>
                <MetricCard
                  title="Completion Rate"
                  value={`${analyticsData.agentMetrics.completionRate}%`}
                  icon={Brain}
                  trend={analyticsData.agentMetrics.completionRate > 95 ? { value: 2.3, isPositive: true } : undefined}
                  color="green"
                />
              </motion.div>
              <motion.div variants={fadeIn}>
                <MetricCard
                  title="Vulnerabilities"
                  value={`${analyticsData.securityMetrics.vulnerabilitiesDetected} Found`}
                  icon={Shield}
                  color="red"
                />
              </motion.div>
            </motion.div>

            {/* Performance Metrics - Centered */}
            <motion.div 
              className="grid grid-cols-2 gap-6 max-w-2xl mx-auto"
              initial="hidden"
              animate="visible"
              variants={{
                visible: {
                  transition: {
                    staggerChildren: 0.1,
                    delayChildren: 0.1
                  }
                }
              }}
            >
              <motion.div variants={fadeIn}>
                <Card className="bg-zinc-900/60 border-zinc-800 backdrop-blur-xl">
                  <CardHeader className="flex flex-row items-center justify-between pb-2">
                    <CardTitle className="text-sm font-medium text-zinc-400">Avg Response Time</CardTitle>
                    <Clock className="h-4 w-4 text-blue-400" />
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-baseline justify-between">
                      <div className="text-3xl font-bold text-white">{analyticsData.overview.avgResponseTime}ms</div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
              <motion.div variants={fadeIn}>
                <Card className="bg-zinc-900/60 border-zinc-800 backdrop-blur-xl">
                  <CardHeader className="flex flex-row items-center justify-between pb-2">
                    <CardTitle className="text-sm font-medium text-zinc-400">Queries/Second</CardTitle>
                    <Zap className="h-4 w-4 text-yellow-400" />
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-baseline justify-between">
                      <div className="text-3xl font-bold text-white">{analyticsData.performance.qps}</div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Recent Activity Timeline - Fixed Height with Internal Scrolling */}
            <motion.div
              initial="hidden"
              animate="visible"
              variants={{
                visible: {
                  transition: {
                    staggerChildren: 0.1,
                    delayChildren: 0.2
                  }
                }
              }}
            >
              <motion.div variants={fadeIn}>
                <Card className="bg-zinc-900/60 border-zinc-800 backdrop-blur-xl h-[32rem] flex flex-col relative overflow-hidden">
                  {/* Card Header */}
                  <CardHeader className="flex-shrink-0 relative z-10 bg-zinc-900/80 backdrop-blur-sm">
                    <CardTitle className="text-lg font-medium text-white flex items-center gap-2">
                      <Activity className="h-5 w-5 text-purple-400" />
                      Latest Activity
                    </CardTitle>
                    <p className="text-sm text-zinc-400">Recent execution timeline</p>
                  </CardHeader>
                  
                  {/* Scrollable Content with Gradients */}
                  <CardContent className="p-0 flex-1 relative overflow-hidden">
                    {/* Top fade gradient */}
                    <div className="absolute top-0 left-0 right-0 h-4 bg-gradient-to-b from-zinc-900/60 to-transparent z-20 pointer-events-none" />
                    
                    {/* Bottom fade gradient */}
                    <div className="absolute bottom-0 left-0 right-0 h-6 bg-gradient-to-t from-zinc-900/80 via-zinc-900/40 to-transparent z-20 pointer-events-none" />
                    
                    {recentActivity.length > 0 ? (
                      <div className="h-full overflow-y-auto space-y-1 p-4 pt-2 pb-8 hide-scrollbar">
                        {recentActivity.map((activity, index) => (
                          <motion.div
                            key={`${activity.type}-${activity.timestamp}-${index}`}
                            variants={fadeIn}
                            className="flex items-start gap-3 p-3 rounded-lg bg-zinc-800/30 hover:bg-zinc-800/50 transition-all duration-200 border border-zinc-700/30 hover:border-zinc-600/40 group relative overflow-hidden"
                          >
                            {/* Subtle gradient overlay on hover */}
                            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                            
                            <div className="flex-shrink-0 mt-1.5 relative z-10">
                              {activity.type === 'log' ? (
                                <div className={`w-2.5 h-2.5 rounded-full shadow-lg transition-all duration-200 ${activity.hasVulnerability ? 'bg-red-400 shadow-red-400/50 group-hover:shadow-red-400/70' : 'bg-blue-400 shadow-blue-400/50 group-hover:shadow-blue-400/70'}`} />
                              ) : (
                                <div className={`w-2.5 h-2.5 rounded-full shadow-lg transition-all duration-200 ${activity.hasVulnerability ? 'bg-orange-400 shadow-orange-400/50 group-hover:shadow-orange-400/70' : 'bg-green-400 shadow-green-400/50 group-hover:shadow-green-400/70'}`} />
                              )}
                            </div>
                            <div className="flex-1 min-w-0 relative z-10">
                              <div className="flex items-center justify-between mb-1.5">
                                <h4 className="text-sm font-medium text-white truncate pr-2 group-hover:text-purple-200 transition-colors">{activity.title}</h4>
                                <span className="text-xs text-zinc-500 flex-shrink-0 font-mono bg-zinc-800/50 px-1.5 py-0.5 rounded group-hover:bg-zinc-700/50 transition-colors">
                                  {new Date(activity.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                              </div>
                              <p className="text-xs text-zinc-400 mb-2 line-clamp-2 leading-relaxed group-hover:text-zinc-300 transition-colors">{activity.description}</p>
                              <div className="flex items-center gap-1.5">
                                <Badge 
                                  variant="outline" 
                                  className={`text-xs font-medium transition-all duration-200 ${activity.type === 'log' ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 group-hover:bg-blue-500/20' : 'bg-purple-500/10 text-purple-300 border-purple-500/30 group-hover:bg-purple-500/20'}`}
                                >
                                  {activity.type}
                                </Badge>
                                {activity.hasVulnerability && (
                                  <Badge variant="outline" className="text-xs font-medium bg-red-500/10 text-red-300 border-red-500/30 group-hover:bg-red-500/20 transition-all duration-200">
                                    vulnerability
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    ) : (
                      <div className="h-full flex items-center justify-center relative z-10">
                        <div className="text-center">
                          <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-zinc-800/50 flex items-center justify-center">
                            <Activity className="w-6 h-6 text-zinc-600" />
                          </div>
                          <p className="text-zinc-500 text-sm">No recent activity</p>
                          <p className="text-zinc-600 text-xs mt-1">Agent executions will appear here</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </motion.div>

            {/* Summary Card */}
            {analyticsData.overview.totalRequests === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-12"
              >
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-purple-500/10 flex items-center justify-center">
                  <LineChart className="w-8 h-8 text-purple-400" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">Waiting for Agent Activity</h3>
                <p className="text-zinc-400 mb-4">
                  No analytics data available yet. Start running your Python agents to see performance metrics.
                </p>
                <div className="text-sm text-zinc-500 bg-zinc-800/50 rounded-lg p-4 max-w-md mx-auto">
                  <p className="mb-2">📊 <strong>What you'll see:</strong></p>
                  <ul className="text-left space-y-1">
                    <li>• Real-time performance metrics</li>
                    <li>• Agent completion rates</li>
                    <li>• Tool usage analytics</li>
                    <li>• Security assessments</li>
                    <li>• Error tracking and trends</li>
                    <li>• Live execution timeline</li>
                  </ul>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
