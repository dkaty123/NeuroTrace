"use client";

import React, { useCallback } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  NodeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';
import WorkflowNode from './WorkflowNode';

const nodeTypes: NodeTypes = {
  workflow: WorkflowNode,
};

// Calculate positions to spread nodes across the space
const VERTICAL_SPACING = 150;
const HORIZONTAL_SPACING = 300;
const CENTER_X = HORIZONTAL_SPACING;

const nodeColorClasses = [
  'bg-sky-700/70 border border-sky-500 text-sky-100',
  'bg-amber-700/70 border border-amber-500 text-amber-100',
  'bg-teal-700/70 border border-teal-500 text-teal-100',
  'bg-rose-700/70 border border-rose-500 text-rose-100',
  'bg-lime-700/70 border border-lime-500 text-lime-100',
  'bg-violet-700/70 border border-violet-500 text-violet-100',
];

const baseNodesData = [
  {
    id: 'query_analysis',
    label: 'Query Analysis',
    description: 'Analyzes the research query and creates a structured research plan'
  },
  {
    id: 'search',
    label: 'Web Research',
    description: 'Conducts web research using search tools and processes results'
  },
  {
    id: 'analyze',
    label: 'Content Analysis',
    description: 'Analyzes processed sources and extracts key findings'
  },
  {
    id: 'synthesize',
    label: 'Synthesis',
    description: 'Synthesizes findings into insights and recommendations'
  },
  {
    id: 'fact_check',
    label: 'Fact Check',
    description: 'Performs fact-checking and quality assessment'
  },
  {
    id: 'generate_report',
    label: 'Generate Report',
    description: 'Generates the final comprehensive research report'
  }
];

const initialNodes: Node[] = baseNodesData.map((nodeData, index) => ({
  id: nodeData.id,
  type: 'workflow',
  data: {
    label: nodeData.label,
    description: nodeData.description
  },
  position: { x: CENTER_X, y: VERTICAL_SPACING * index },
  className: nodeColorClasses[index % nodeColorClasses.length]
}));

const initialEdges: Edge[] = [
  { 
    id: 'e1-2', 
    source: 'query_analysis', 
    target: 'search', 
    animated: true,
    style: { stroke: '#fff' },
  },
  { 
    id: 'e2-3', 
    source: 'search', 
    target: 'analyze', 
    animated: true,
    style: { stroke: '#fff' },
  },
  { 
    id: 'e3-4', 
    source: 'analyze', 
    target: 'synthesize', 
    animated: true,
    style: { stroke: '#fff' },
  },
  { 
    id: 'e4-5', 
    source: 'synthesize', 
    target: 'fact_check', 
    animated: true,
    style: { stroke: '#fff' },
  },
  { 
    id: 'e5-6', 
    source: 'fact_check', 
    target: 'generate_report', 
    animated: true,
    style: { stroke: '#fff' },
  }
];

export default function WorkflowGraph() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    console.log('clicked node:', node);
  }, []);

  return (
    <div className="w-full h-full bg-zinc-900 rounded-lg overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={1.5}
        attributionPosition="bottom-left"
        defaultEdgeOptions={{
          type: 'smoothstep',
          style: { stroke: '#fff' },
        }}
        style={{ background: '#18181b' }}
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
          className="!bg-zinc-950 !border !border-purple-500/50 !rounded-md"
        />
      </ReactFlow>
    </div>
  );
} 