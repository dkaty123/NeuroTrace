"use client";

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, AlertTriangle, Shield, Code, FileText, Eye, EyeOff } from 'lucide-react';
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface CodeVulnerability {
  type: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  line?: number;
  suggestion: string;
}

interface AgentCodeEntry {
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

interface AgentDetailsProps {
  agent: AgentCodeEntry | null;
}

const severityColors = {
  critical: 'bg-red-500/10 text-red-400 border-red-500/20',
  high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  low: 'bg-blue-500/10 text-blue-400 border-blue-500/20'
};

const severityIcons = {
  critical: '🔴',
  high: '🟠',
  medium: '🟡',
  low: '🔵'
};

export default function AgentDetails({ agent }: AgentDetailsProps) {
  const [expandedVulns, setExpandedVulns] = useState<string[]>([]);
  const [showingCode, setShowingCode] = useState<string[]>([]);

  if (!agent) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500">
        <div className="text-center">
          <Shield className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>Select an agent to view security analysis</p>
        </div>
      </div>
    );
  }

  const toggleVuln = (vulnId: string) => {
    setExpandedVulns(prev => 
      prev.includes(vulnId) 
        ? prev.filter(id => id !== vulnId)
        : [...prev, vulnId]
    );
  };

  const toggleCode = (vulnId: string) => {
    setShowingCode(prev =>
      prev.includes(vulnId)
        ? prev.filter(id => id !== vulnId)
        : [...prev, vulnId]
    );
  };

  const vulnCount = agent.vulnerabilities?.length || 0;
  const criticalCount = agent.vulnerabilities?.filter(v => v.type === 'critical').length || 0;
  const highCount = agent.vulnerabilities?.filter(v => v.type === 'high').length || 0;

  return (
    <div className="h-full flex flex-col bg-zinc-900">
      {/* Header */}
      <div className="border-b border-zinc-800 p-4">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
            <Code className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">{agent.agent_name}</h2>
            <p className="text-sm text-zinc-400">{agent.description || 'No description available'}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1.5 text-zinc-400">
            <FileText className="w-4 h-4" />
            <span>{agent.file_path || 'Unknown path'}</span>
          </div>
          <div className="flex items-center gap-1.5 text-zinc-400">
            <AlertTriangle className="w-4 h-4" />
            <span>{vulnCount} {vulnCount === 1 ? 'vulnerability' : 'vulnerabilities'}</span>
          </div>
        </div>
      </div>

      {/* Vulnerability Summary */}
      <div className="border-b border-zinc-800 bg-black/20 px-4 py-3">
        <div className="flex items-center gap-3">
          {criticalCount > 0 && (
            <Badge className={severityColors.critical}>
              {criticalCount} Critical
            </Badge>
          )}
          {highCount > 0 && (
            <Badge className={severityColors.high}>
              {highCount} High
            </Badge>
          )}
        </div>
      </div>

      {/* Vulnerabilities List */}
      <div className="flex-1 overflow-y-auto">
        {agent.vulnerabilities?.length ? (
          <div className="divide-y divide-zinc-800">
            {agent.vulnerabilities.map((vuln, index) => {
              const vulnId = `${agent.agent_id}-${index}`;
              const isExpanded = expandedVulns.includes(vulnId);
              const isShowingCode = showingCode.includes(vulnId);
              
              return (
                <div key={vulnId} className="group">
                  <div className="px-4 py-3 hover:bg-white/5 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <span className="text-lg" role="img" aria-label={vuln.type}>
                          {severityIcons[vuln.type]}
                        </span>
                        <div>
                          <h3 className="font-medium text-white group-hover:text-purple-400 transition-colors">
                            {vuln.title}
                          </h3>
                          <p className="text-sm text-zinc-400">
                            {vuln.description}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleCode(vulnId);
                          }}
                          className="h-7 px-2 text-xs hover:bg-zinc-800"
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
                        <button
                          onClick={() => toggleVuln(vulnId)}
                          className="p-1 hover:bg-white/10 rounded transition-colors"
                        >
                          {isExpanded ? (
                            <ChevronDown className="w-5 h-5 text-zinc-400" />
                          ) : (
                            <ChevronRight className="w-5 h-5 text-zinc-400" />
                          )}
                        </button>
                      </div>
                    </div>
                    
                    {/* Code Preview */}
                    {vuln.line && isShowingCode && (
                      <div className="mt-2 p-2 rounded bg-black/50 border border-zinc-800 font-mono text-sm">
                        <div className="flex items-center justify-between text-xs text-zinc-500 mb-1 text-wrap">
                          <span>Line {vuln.line}</span>
                        </div>
                        <pre className="text-zinc-300 text-wrap">
                          {agent.source_code.split('\n')[vuln.line - 1]}
                        </pre>
                      </div>
                    )}
                  </div>
                  
                  {/* Expanded Details */}
                  {isExpanded && (
                    <div className="px-4 pb-4">
                      <div className="rounded-lg bg-black/30 border border-zinc-800 p-4 space-y-3">
                        <div>
                          <h4 className="text-sm font-medium text-zinc-300 mb-1">Suggestion</h4>
                          <p className="text-sm text-zinc-400">{vuln.suggestion}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-4 text-center text-zinc-500">
            <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No vulnerabilities detected</p>
          </div>
        )}
      </div>
    </div>
  );
} 