import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';
import readline from 'readline';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface NodeVulnerability {
  type: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  description: string;
  evidence: string;
  recommendation: string;
  code_location?: string;
}

interface NodeAnalysis {
  node_name: string;
  function_name: string;
  vulnerabilities: NodeVulnerability[];
  thinking_patterns: Array<{
    pattern: string;
    risk_level: string;
    description: string;
  }>;
  security_score: number;
}

async function fetchLogsAndNodeFunctions(): Promise<{logs: any[], nodeFunctions: any}> {
  const logsDir = path.join(process.cwd(), '..', 'be', 'logs');
  
  if (!fs.existsSync(logsDir)) {
    throw new Error('Logs directory not found');
  }
  
  // Get the most recent files
  const jsonlFiles = fs.readdirSync(logsDir)
    .filter(file => file.endsWith('.jsonl'))
    .sort((a, b) => {
      const statA = fs.statSync(path.join(logsDir, a));
      const statB = fs.statSync(path.join(logsDir, b));
      return statB.mtime.getTime() - statA.mtime.getTime();
    });
  
  const jsonFiles = fs.readdirSync(logsDir)
    .filter(file => file.startsWith('node_functions_') && file.endsWith('.json'))
    .sort((a, b) => {
      const statA = fs.statSync(path.join(logsDir, a));
      const statB = fs.statSync(path.join(logsDir, b));
      return statB.mtime.getTime() - statA.mtime.getTime();
    });
  
  if (jsonlFiles.length === 0) {
    throw new Error('No execution logs found');
  }
  
  // Read execution logs
  const logs: any[] = [];
  const latestJsonlFile = jsonlFiles[0];
  const jsonlPath = path.join(logsDir, latestJsonlFile);
  
  const fileStream = fs.createReadStream(jsonlPath);
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity
  });
  
  for await (const line of rl) {
    if (line.trim()) {
      try {
        const logEntry = JSON.parse(line);
        logs.push(logEntry);
      } catch (parseError) {
        console.error('Error parsing log line:', parseError);
      }
    }
  }
  
  // Read node functions
  let nodeFunctions = {};
  if (jsonFiles.length > 0) {
    const latestJsonFile = jsonFiles[0];
    const jsonPath = path.join(logsDir, latestJsonFile);
    const nodeData = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
    nodeFunctions = nodeData.nodes || {};
  }
  
  return { logs, nodeFunctions };
}

export async function GET(request: NextRequest) {
  try {
    // Check for OpenAI API key
    if (!process.env.OPENAI_API_KEY) {
      return NextResponse.json(
        { error: 'OpenAI API key not configured. Please set OPENAI_API_KEY in .env.local' },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(request.url);
    const severity = searchParams.get('severity') || 'all'; // 'all', 'high', 'critical'
    
    // Fetch logs and node function source code
    const { logs, nodeFunctions } = await fetchLogsAndNodeFunctions();
    
    if (logs.length === 0) {
      return NextResponse.json({ error: 'No logs available for analysis' }, { status: 404 });
    }
    
    // Group logs by node
    const nodeData: Record<string, any> = {};
    
    logs.forEach(log => {
      const node = log.current_node || 'global';
      if (!nodeData[node]) {
        nodeData[node] = {
          execution_logs: [],
          thinking_logs: [],
          errors: [],
          prompts: [],
          responses: []
        };
      }
      
      nodeData[node].execution_logs.push(log);
      
      if (log.event === 'llm_thinking_start' && log.prompt) {
        nodeData[node].prompts.push(log.prompt);
      }
      if (log.event === 'llm_thinking_end' && log.response) {
        nodeData[node].responses.push(log.response);
      }
      if (log.error) {
        nodeData[node].errors.push(log.error);
      }
      if (log.event.includes('thinking')) {
        nodeData[node].thinking_logs.push(log);
      }
    });
    
    // Prepare all nodes for batch analysis
    const nodeEntries = Object.entries(nodeFunctions);
    
    // Build batch analysis prompt with all nodes
    const nodesInfo = nodeEntries.map(([nodeName, nodeFunction]) => {
      const data = nodeData[nodeName] || {
        execution_logs: [],
        thinking_logs: [],
        errors: [],
        prompts: [],
        responses: []
      };
      
      const functionSource = (nodeFunction as any)?.source_code || 'Source code not available';
      const functionName = (nodeFunction as any)?.metadata?.name || nodeName;
      
      // Truncate source code for speed
      const truncatedSource = functionSource.length > 1000 ? 
        functionSource.substring(0, 1000) + '\n... [truncated]' : 
        functionSource;
      
      return {
        name: nodeName,
        function_name: functionName,
        source: truncatedSource,
        prompts: data.prompts.length,
        responses: data.responses.length,
        errors: data.errors.length,
        sample_prompt: data.prompts[0]?.substring(0, 150) || 'None',
        sample_error: data.errors[0]?.substring(0, 100) || 'None'
      };
    }).slice(0, 6); // Limit to 6 nodes for speed
    
    const batchPrompt = `Analyze these ${nodesInfo.length} LangGraph nodes for vulnerabilities. Return JSON array with analysis for each node.

NODES:
${nodesInfo.map((node, i) => `
${i + 1}. ${node.name} (${node.function_name})
CODE: ${node.source.substring(0, 500)}...
DATA: ${node.prompts} prompts, ${node.responses} responses, ${node.errors} errors
SAMPLE: ${node.sample_prompt}
ERROR: ${node.sample_error}
`).join('')}

Return ONLY this JSON format:
[
  {
    "node_name": "${nodesInfo[0]?.name}",
    "vulnerabilities": [{"type": "Input Validation", "severity": "HIGH", "description": "Issue", "evidence": "Code evidence", "recommendation": "Fix"}],
    "thinking_patterns": [{"pattern": "Risk pattern", "risk_level": "MEDIUM", "description": "Why risky"}],
    "security_score": 75
  }
]

Focus on real vulnerabilities. Be concise.`;

    let nodeAnalyses: NodeAnalysis[] = [];
    
    try {
      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'Security expert. Return ONLY valid JSON array. No explanations.'
          },
          {
            role: 'user',
            content: batchPrompt
          }
        ],
        temperature: 0.3, // Slightly higher for faster generation
        max_tokens: 1500 // Reduced for speed
      });
      
      const analysisText = response.choices[0]?.message?.content;
      
      if (!analysisText) {
        throw new Error('No analysis generated');
      }
      
      // Parse batch analysis
      let batchAnalysis;
      try {
        batchAnalysis = JSON.parse(analysisText);
        if (!Array.isArray(batchAnalysis)) {
          throw new Error('Expected array response');
        }
      } catch (parseError) {
        console.error('Batch analysis parse error:', parseError);
        // Fallback to individual analysis for critical nodes
        batchAnalysis = nodesInfo.map(node => ({
          node_name: node.name,
          vulnerabilities: [{
            type: 'Analysis Error',
            severity: 'MEDIUM',
            description: 'Batch analysis failed',
            evidence: 'JSON parsing error',
            recommendation: 'Retry analysis'
          }],
          thinking_patterns: [],
          security_score: 50
        }));
      }
      
      // Convert to NodeAnalysis format
      nodeAnalyses = batchAnalysis.map((analysis: any, index: number) => {
        const nodeInfo = nodesInfo[index];
        
        const nodeAnalysis: NodeAnalysis = {
          node_name: analysis.node_name || nodeInfo?.name || `node_${index}`,
          function_name: nodeInfo?.function_name || analysis.node_name || `func_${index}`,
          vulnerabilities: analysis.vulnerabilities || [],
          thinking_patterns: analysis.thinking_patterns || [],
          security_score: analysis.security_score || 50
        };
        
        // Filter by severity if requested
        if (severity !== 'all') {
          const severityFilter = severity.toUpperCase();
          nodeAnalysis.vulnerabilities = nodeAnalysis.vulnerabilities.filter((v: any) => 
            severityFilter === 'HIGH' ? ['HIGH', 'CRITICAL'].includes(v.severity) :
            severityFilter === 'CRITICAL' ? v.severity === 'CRITICAL' :
            true
          );
        }
        
        return nodeAnalysis;
      });
      
    } catch (error: any) {
      console.error('Batch analysis error:', error);
      
      // Fallback: Quick parallel analysis of critical nodes only
      const criticalNodes = nodeEntries.slice(0, 3); // Only analyze first 3 nodes
      
      const parallelAnalyses = await Promise.allSettled(
        criticalNodes.map(async ([nodeName, nodeFunction]) => {
          const data = nodeData[nodeName] || {
            execution_logs: [], thinking_logs: [], errors: [], prompts: [], responses: []
          };
          
          const functionSource = (nodeFunction as any)?.source_code || 'Source code not available';
          const functionName = (nodeFunction as any)?.metadata?.name || nodeName;
          
          // Quick mini-prompt for speed
          const quickPrompt = `Analyze ${nodeName} for vulnerabilities:
CODE: ${functionSource.substring(0, 300)}...
DATA: ${data.prompts.length} prompts, ${data.errors.length} errors

Return JSON: {"vulnerabilities": [{"type":"Type", "severity":"HIGH", "description":"Issue", "evidence":"Evidence", "recommendation":"Fix"}], "security_score": 80}`;
          
          try {
            const response = await openai.chat.completions.create({
              model: 'gpt-4o-mini',
              messages: [
                { role: 'system', content: 'Return only JSON.' },
                { role: 'user', content: quickPrompt }
              ],
              temperature: 0.5,
              max_tokens: 500
            });
            
            const analysisText = response.choices[0]?.message?.content;
            let analysis;
            
            try {
              analysis = JSON.parse(analysisText || '{}');
            } catch {
              analysis = { vulnerabilities: [], security_score: 70 };
            }
            
            return {
              node_name: nodeName,
              function_name: functionName,
              vulnerabilities: analysis.vulnerabilities || [],
              thinking_patterns: [],
              security_score: analysis.security_score || 70
            };
            
          } catch (err) {
            return {
              node_name: nodeName,
              function_name: functionName,
              vulnerabilities: [{
                type: 'Analysis Error',
                severity: 'LOW',
                description: 'Quick analysis failed',
                evidence: 'API error',
                recommendation: 'Retry later'
              }],
              thinking_patterns: [],
              security_score: 60
            };
          }
        })
      );
      
             nodeAnalyses = parallelAnalyses.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          const nodeName = criticalNodes[index][0];
          return {
            node_name: nodeName,
            function_name: nodeName,
            vulnerabilities: [{
              type: 'Analysis Failed',
              severity: 'MEDIUM',
              description: 'Could not analyze node',
              evidence: 'Promise rejected',
              recommendation: 'Check logs and retry'
            }],
            thinking_patterns: [],
            security_score: 50
          };
        }
      });
    }
    
    // Calculate summary metrics
    const totalVulnerabilities = nodeAnalyses.reduce((sum, node) => sum + node.vulnerabilities.length, 0);
    const averageScore = nodeAnalyses.reduce((sum, node) => sum + node.security_score, 0) / nodeAnalyses.length;
    const criticalVulns = nodeAnalyses.reduce((sum, node) => 
      sum + node.vulnerabilities.filter(v => v.severity === 'CRITICAL').length, 0);
    const highVulns = nodeAnalyses.reduce((sum, node) => 
      sum + node.vulnerabilities.filter(v => v.severity === 'HIGH').length, 0);
    
    return NextResponse.json({
      summary: {
        total_nodes: nodeAnalyses.length,
        total_vulnerabilities: totalVulnerabilities,
        critical_vulnerabilities: criticalVulns,
        high_vulnerabilities: highVulns,
        average_security_score: Math.round(averageScore),
        analysis_timestamp: new Date().toISOString(),
        logs_analyzed: logs.length
      },
      nodes: nodeAnalyses,
      immediate_actions: nodeAnalyses
        .flatMap(n => n.vulnerabilities.filter(v => ['CRITICAL', 'HIGH'].includes(v.severity)))
        .slice(0, 5)
        .map(v => v.recommendation)
    });
    
  } catch (error: any) {
    console.error('Analysis error:', error);
    
    if (error.message?.includes('Logs directory not found')) {
      return NextResponse.json(
        { error: 'Backend logs not accessible. Run a workflow first.' },
        { status: 404 }
      );
    }
    
    if (error.code === 'invalid_api_key') {
      return NextResponse.json(
        { error: 'Invalid OpenAI API key. Check your .env.local file.' },
        { status: 401 }
      );
    }
    
    return NextResponse.json(
      { error: 'Analysis failed', details: error.message },
      { status: 500 }
    );
  }
} 