"use client";

import React, { useCallback, useEffect, useState, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  NodeTypes,
  EdgeTypes,
  MarkerType,
  ConnectionLineType,
  Handle,
  Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Code, Clock, FileText, Cpu } from 'lucide-react';

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

// Custom node component for agent code
const AgentCodeNode = ({ data, selected }: { data: any; selected?: boolean }) => {
  const { label, description, source_code, dependencies, agent_id } = data;
  const lineCount = source_code.split('\n').length;
  const isSelected = selected || false;
  
  return (
    <div className={`
      relative px-5 py-4 rounded-xl border-2 bg-gradient-to-br ${data.className} 
      min-w-[220px] max-w-[280px] w-[250px]
      ${isSelected 
        ? 'scale-[1.05] -translate-y-2 shadow-2xl shadow-white/30 border-white ring-2 ring-white/60 ring-offset-2 ring-offset-zinc-900' 
        : 'hover:scale-[1.02] hover:shadow-2xl hover:shadow-purple-500/20 hover:-translate-y-1 hover:border-opacity-100'
      }
      transition-all duration-300 ease-out cursor-pointer
      backdrop-blur-sm shadow-lg
      group overflow-hidden
    `}>
      <Handle
        type="target"
        position={Position.Top}
        id="top"
        className="!bg-purple-500 !w-3 !h-3 !border-2 !border-purple-300"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="bottom"
        className="!bg-purple-500 !w-3 !h-3 !border-2 !border-purple-300"
      />
      
      {/* Subtle shine effect */}
      <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -skew-x-12 transition-transform duration-700 ease-out ${
        isSelected ? 'translate-x-full' : '-translate-x-full group-hover:translate-x-full'
      }`} />
      
      {/* Selected indicator glow */}
      {isSelected && (
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/20 via-transparent to-purple-500/20 rounded-xl animate-pulse" />
      )}
      
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="p-1.5 rounded-lg bg-white/10 backdrop-blur-sm">
            <Code className="w-3.5 h-3.5 flex-shrink-0" />
          </div>
          <div className="font-semibold text-sm truncate">{label}</div>
        </div>
        <div className={`flex-shrink-0 w-2 h-2 rounded-full transition-all duration-300 ${
          isSelected ? 'bg-white ring-2 ring-white/50 animate-pulse' : 'bg-green-400 animate-pulse'
        }`} />
      </div>
      
      {/* Description */}
      {description && (
        <div className="text-xs opacity-85 mb-3 leading-relaxed">
          <div className="line-clamp-2 min-h-[2.5rem]">
            {description}
          </div>
        </div>
      )}
      
      {/* Stats */}
      <div className="space-y-2 mb-3">
        <div className="flex items-center gap-2 text-xs">
          <FileText className="w-3 h-3 opacity-60" />
          <span className="opacity-70">
            {lineCount} {lineCount === 1 ? 'line' : 'lines'}
          </span>
        </div>
        
        {dependencies && dependencies.length > 0 && (
          <div className="flex items-start gap-2 text-xs">
            <Cpu className="w-3 h-3 opacity-60 mt-0.5 flex-shrink-0" />
            <div className="opacity-70 leading-tight">
              <span className="text-xs font-medium">deps:</span>{' '}
              <span className="break-words">
                {dependencies.slice(0, 3).join(', ')}
                {dependencies.length > 3 ? '...' : ''}
              </span>
            </div>
          </div>
        )}
      </div>
      
      {/* Action hint */}
      <div className="text-xs opacity-50 text-center py-1 border-t border-white/10 bg-black/20 -mx-5 -mb-4 mt-3 group-hover:opacity-70 transition-opacity duration-200">
        <Clock className="w-3 h-3 inline mr-1" />
        Click to explore
      </div>
    </div>
  );
};

const nodeTypes: NodeTypes = {
  agentCode: (nodeProps: any) => <AgentCodeNode {...nodeProps} selected={nodeProps.data.selected} />,
};

const edgeTypes: EdgeTypes = {
  // Use default edge types
};

const nodeColorClasses = [
  'border-blue-500 from-blue-900/40 to-blue-800/60 text-blue-100',
  'border-green-500 from-green-900/40 to-green-800/60 text-green-100',
  'border-purple-500 from-purple-900/40 to-purple-800/60 text-purple-100',
  'border-yellow-500 from-yellow-900/40 to-yellow-800/60 text-yellow-100',
  'border-pink-500 from-pink-900/40 to-pink-800/60 text-pink-100',
  'border-cyan-500 from-cyan-900/40 to-cyan-800/60 text-cyan-100',
  'border-orange-500 from-orange-900/40 to-orange-800/60 text-orange-100',
  'border-indigo-500 from-indigo-900/40 to-indigo-800/60 text-indigo-100',
];

interface AgentCodeFlowProps {
  onNodeSelect?: (agent: AgentCodeEntry | null) => void;
}

export default function AgentCodeFlow({ onNodeSelect }: AgentCodeFlowProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [agentData, setAgentData] = useState<AgentCodeEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const agentDataRef = useRef<AgentCodeEntry[]>([]);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Update ref whenever agentData changes
  useEffect(() => {
    agentDataRef.current = agentData;
  }, [agentData]);

  const fetchAgentData = useCallback(async (isInitialLoad = false) => {
    if (!mounted) return;

    try {
      if (isInitialLoad) setLoading(true);

      const response = await fetch('/api/get-agent-code', {
        cache: 'no-store'
      });

      if (!response.ok) {
        console.error('Failed to fetch agent code data');
        return;
      }

      const data: AgentCodeEntry[] = await response.json();
      console.log('🔍 Fetched agent data:', data.length, 'agents', data);
      
      // Only update if there are actually new agents
      const currentData = agentDataRef.current;
      const hasNewAgents = data.length !== currentData.length || 
        (data.length > 0 && currentData.length > 0 && 
         data[data.length - 1].processed_at !== currentData[currentData.length - 1]?.processed_at);

      if (hasNewAgents || isInitialLoad) {
        console.log(`🔄 Setting agent data: ${data.length} agents (${hasNewAgents ? 'new data' : 'initial load'})`);
        setAgentData(data);
      } else {
        console.log('⏭️ No new agents, skipping update');
      }
      
    } catch (error) {
      console.error('Error fetching agent data:', error);
    } finally {
      if (isInitialLoad) setLoading(false);
    }
  }, [mounted]);

  useEffect(() => {
    if (!mounted) return;

    // Initial load
    fetchAgentData(true);

    // Poll for updates every 5 seconds
    const interval = setInterval(() => fetchAgentData(false), 5000);
    return () => clearInterval(interval);
  }, [mounted, fetchAgentData]);

  // Generate nodes and edges from agent data
  useEffect(() => {
    console.log('🔄 Generating flow from agent data:', agentData.length, 'agents');
    if (!agentData.length) {
      console.log('❌ No agent data, skipping flow generation');
      setNodes([]); // Clear nodes if no agent data
      setEdges([]); // Clear edges if no agent data
      return;
    }

    const VERTICAL_SPACING = 250;
    const HORIZONTAL_SPACING = 450;
    const BASE_X = 300;

    // Sort all agents by timestamp
    const sortedAgents = [...agentData].sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    console.log('📊 All agents sorted by timestamp:', sortedAgents.map(a => a.agent_name));

    const newNodes: Node[] = [];
    const newEdges: Edge[] = [];

    // Determine the number of columns for a more spread-out layout if many nodes
    const MAX_NODES_PER_COLUMN = 5;
    const numColumns = Math.ceil(sortedAgents.length / MAX_NODES_PER_COLUMN);

    sortedAgents.forEach((agent, globalIndex) => {
      const columnIndex = Math.floor(globalIndex / MAX_NODES_PER_COLUMN);
      const indexInColumn = globalIndex % MAX_NODES_PER_COLUMN;

      const nodeX = BASE_X + (columnIndex * HORIZONTAL_SPACING);
      const nodeY = VERTICAL_SPACING * indexInColumn;

      const node: Node = {
        id: agent.agent_id,
        type: 'agentCode',
        data: {
          label: agent.agent_name,
          description: agent.description,
          source_code: agent.source_code,
          dependencies: agent.dependencies,
          agent_id: agent.agent_id,
          className: nodeColorClasses[globalIndex % nodeColorClasses.length],
          selected: selectedNodeId === agent.agent_id
        },
        position: { x: nodeX, y: nodeY },
      };
      newNodes.push(node);
      console.log(`🔷 Created node: ${agent.agent_name} at (${nodeX}, ${nodeY})`);

      // Create edge to the immediately previous agent in the sorted list
      if (globalIndex > 0) {
        const previousAgent = sortedAgents[globalIndex - 1];
        const edge: Edge = {
          id: `edge-${previousAgent.agent_id}-to-${agent.agent_id}`,
          source: previousAgent.agent_id,
          target: agent.agent_id,
          type: 'smoothstep',
          animated: true,
          style: {
            stroke: '#a855f7',
            strokeWidth: 3,
            strokeOpacity: 0.8
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#a855f7',
            width: 20,
            height: 20
          },
          zIndex: 1000,
          sourceHandle: 'bottom',
          targetHandle: 'top',
          data: {
            sourceNode: previousAgent.agent_name,
            targetNode: agent.agent_name
          }
        };
        newEdges.push(edge);
        console.log(`📎 Created edge: ${previousAgent.agent_name} (${previousAgent.agent_id}) -> ${agent.agent_name} (${agent.agent_id})`);
      }
    });

    console.log('✅ Generated flow:', { nodes: newNodes.length, edges: newEdges.length });
    console.log('🔍 Node IDs:', newNodes.map(n => n.id));
    console.log('🔍 Edges full:', newEdges);
    
    // Verify edge integrity
    newEdges.forEach(edge => {
      const sourceExists = newNodes.some(n => n.id === edge.source);
      const targetExists = newNodes.some(n => n.id === edge.target);
      console.log(`Edge ${edge.id}: source ${edge.source} exists: ${sourceExists}, target ${edge.target} exists: ${targetExists}`);
    });
    
    setNodes(newNodes);
    setEdges(newEdges);
  }, [agentData, selectedNodeId, setNodes, setEdges]);

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    console.log('clicked agent node:', node);
    
    // Set selected node
    setSelectedNodeId(node.id);
    
    // Find the actual agent data from agentData array
    const currentAgent = agentData.find(agent => agent.agent_id === node.data.agent_id);
    
    if (currentAgent) {
      onNodeSelect?.(currentAgent);
    }
    
    // Subtle zoom focus on selected node
    setTimeout(() => {
      const nodeElement = document.querySelector(`[data-id="${node.id}"]`);
      if (nodeElement) {
        nodeElement.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center', 
          inline: 'center' 
        });
      }
    }, 100);
  }, [onNodeSelect, agentData]);

  const onPaneClick = useCallback(() => {
    console.log('clicked on background - deselecting node');
    
    // Clear selection
    setSelectedNodeId(null);
    onNodeSelect?.(null);
  }, [onNodeSelect]);

  if (!mounted) {
    return (
      <div className="w-full h-full bg-zinc-900 rounded-lg overflow-hidden flex items-center justify-center">
        <div className="text-gray-400">Loading agent flow...</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="w-full h-full bg-zinc-900 rounded-lg overflow-hidden flex items-center justify-center">
        <div className="flex items-center gap-2 text-gray-400">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
          Loading agent code flow...
        </div>
      </div>
    );
  }

  if (agentData.length === 0) {
    return (
      <div className="w-full h-full bg-zinc-900 rounded-lg overflow-hidden flex items-center justify-center">
        <div className="text-center text-gray-400">
          <Code className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>No agent code available yet</p>
          <p className="text-sm">Run a workflow to see agents here</p>
          <p className="text-xs mt-2">Fetching from: /api/get-agent-code</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-zinc-900 rounded-lg overflow-hidden relative">
      {/* Debug info panel */}
      <div className="absolute top-4 right-4 z-10 bg-black/90 backdrop-blur-sm text-white text-xs p-3 rounded-lg border border-purple-500/30 shadow-lg max-w-xs">
        <div className="flex items-center gap-2 mb-2 text-purple-300 font-medium">
          <Cpu className="w-3 h-3" />
          Flow Status
        </div>
        <div className="space-y-1 text-gray-300">
          <div>Agents: <span className="text-green-400 font-mono">{agentData.length}</span></div>
          <div>Nodes: <span className="text-blue-400 font-mono">{nodes.length}</span></div>
          <div>Edges: <span className="text-purple-400 font-mono">{edges.length}</span></div>
        </div>
        {edges.length > 0 && (
          <div className="mt-2 pt-2 border-t border-purple-500/20">
            <div className="text-xs text-gray-400 mb-1">Connections:</div>
            <div className="text-green-400 font-mono text-xs break-all">
              {edges.map(e => `${e.source.slice(-6)}→${e.target.slice(-6)}`).join(' ')}
            </div>
          </div>
        )}
      </div>
      
      <ReactFlow
        key={`flow-${nodes.length}-${edges.length}`}
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionLineType={ConnectionLineType.SmoothStep}
        fitView
        fitViewOptions={{ 
          padding: 0.3,
          includeHiddenNodes: true,
          minZoom: 0.2,
          maxZoom: 2
        }}
        minZoom={0.2}
        maxZoom={2}
        attributionPosition="bottom-left"
        defaultEdgeOptions={{
          type: 'smoothstep',
          style: { 
            stroke: '#a855f7', 
            strokeWidth: 3,
            strokeOpacity: 0.8
          },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#a855f7',
            width: 20,
            height: 20,
          },
          animated: true,
          zIndex: 1000
        }}
        style={{ background: '#18181b' }}
        elementsSelectable={true}
        nodesDraggable={true}
        nodesConnectable={false}
        edgesFocusable={true}
        connectOnClick={false}
        deleteKeyCode={null}
        proOptions={{ hideAttribution: true }}
      >
        <Background 
          color="#3f3f46"
          gap={16} 
          size={1}
          style={{ background: '#18181b' }}
        />
        <Controls className="!bg-zinc-950 !border !border-purple-500/50" />
        <MiniMap 
          style={{ backgroundColor: '#09090b' }}
          nodeColor="#a1a1aa"
          nodeStrokeWidth={3}
          className="!bg-zinc-950 !border !border-purple-500/50 !rounded-md"
        />
      </ReactFlow>
    </div>
  );
} 