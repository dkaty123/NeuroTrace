"use client";

import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';

interface WorkflowNodeProps {
  data: {
    label: string;
    description: string;
  };
  isConnectable: boolean;
}

export default memo(function WorkflowNode({ data, isConnectable }: WorkflowNodeProps) {
  return (
    <div className="px-4 py-2 shadow-lg rounded-md w-[200px]">
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
        className="!bg-white"
      />
      <div className="flex flex-col gap-1">
        <div className="font-semibold">{data.label}</div>
        <div className="text-xs opacity-80">{data.description}</div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={isConnectable}
        className="!bg-white"
      />
    </div>
  );
}); 