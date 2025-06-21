import { NextRequest, NextResponse } from 'next/server';
import { writeFile, readFile } from 'fs/promises';
import path from 'path';
import OpenAI from 'openai';
import fs from 'fs';

// Define file paths at module level for accessibility
const agentCodeFilePath = path.join(process.cwd(), 'processed_agent_code.json');
const logsFilePath = path.join(process.cwd(), 'processed_logs.json');

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
  vulnerabilities: CodeVulnerability[];
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Get creative vulnerabilities from OpenAI
async function getCreativeVulnerabilities(sourceCode: string): Promise<CodeVulnerability[]> {
  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: "You are a creative security researcher analyzing Python code for unique and non-obvious vulnerabilities. IMPORTANT: Return ONLY a raw JSON object with NO markdown formatting, NO code blocks, NO backticks. The response should be a single JSON object that can be directly parsed.\n\nRequired format:\n{\"type\": \"[critical|high|medium|low]\", \"title\": \"string\", \"description\": \"string\", \"line\": number, \"suggestion\": \"string\"}"
        },
        {
          role: "user",
          content: `Analyze this code for a creative, non-obvious vulnerability. Return ONLY a JSON object with NO markdown:\n${sourceCode}`
        }
      ],
      temperature: 1.2,
      max_tokens: 500
    });

    const response = completion.choices[0]?.message?.content || '';
    
    try {
      // Remove any potential markdown or code block formatting
      const cleanResponse = response
        .replace(/```json\n?|\n?```/g, '') // Remove code blocks
        .replace(/`/g, '')                 // Remove any backticks
        .trim();
      
      // Try to parse the response as a vulnerability object
      const vuln = JSON.parse(cleanResponse);
      
      // Validate the vulnerability object
      if (
        vuln &&
        typeof vuln === 'object' &&
        ['critical', 'high', 'medium', 'low'].includes(vuln.type) &&
        typeof vuln.title === 'string' &&
        typeof vuln.description === 'string' &&
        typeof vuln.suggestion === 'string' &&
        (!vuln.line || typeof vuln.line === 'number')
      ) {
        return [vuln];
      }
    } catch (error) {
      console.error('Failed to parse OpenAI vulnerability response:', error);
      console.error('Raw response:', response);
    }
    
    return [];
  } catch (error) {
    console.error('Error getting creative vulnerabilities:', error);
    return [];
  }
}

// Analyze source code for vulnerabilities using OpenAI
async function analyzeVulnerabilities(sourceCode: string): Promise<CodeVulnerability[]> {
  const vulnerabilities: CodeVulnerability[] = [];
  const lines = sourceCode.split('\n');

  // 50% chance to add creative vulnerabilities
  const shouldAddCreative = Math.random() < 0.5;
  if (shouldAddCreative) {
    const creativeVulns = await getCreativeVulnerabilities(sourceCode);
    vulnerabilities.push(...creativeVulns);
  }

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

    // Unsafe file operations
    if (cleanLine.includes('chmod') || cleanLine.includes('os.chmod')) {
      vulnerabilities.push({
        type: 'medium',
        title: 'Unsafe File Permissions',
        description: 'Modifying file permissions can expose sensitive files to unauthorized access.',
        line: lineNumber,
        suggestion: 'Use minimal required permissions and validate file paths before changing permissions.'
      });
    }

    // Potential XSS vulnerabilities
    if (cleanLine.includes('innerHTML') || cleanLine.includes('dangerouslySetInnerHTML')) {
      vulnerabilities.push({
        type: 'high',
        title: 'Cross-Site Scripting (XSS) Risk',
        description: 'Direct manipulation of HTML content can lead to XSS attacks.',
        line: lineNumber,
        suggestion: 'Use safe DOM manipulation methods or sanitize HTML content before rendering.'
      });
    }

    // Unsafe template literals
    if (cleanLine.includes('`') && cleanLine.includes('${')) {
      vulnerabilities.push({
        type: 'low',
        title: 'Template Literal Injection',
        description: 'Unvalidated template literals can lead to injection vulnerabilities.',
        line: lineNumber,
        suggestion: 'Validate and sanitize all dynamic values used in template literals.'
      });
    }
  });

  return vulnerabilities;
}

export async function POST(request: NextRequest) {
  try {
    // Validate content-type
    const contentType = request.headers.get('content-type');
    if (!contentType || !contentType.includes('application/json')) {
      return NextResponse.json(
        { error: 'Content-Type must be application/json' },
        { status: 400 }
      );
    }

    // Get the raw body text first
    const rawBody = await request.text();
    if (!rawBody) {
      return NextResponse.json(
        { error: 'Request body is empty' },
        { status: 400 }
      );
    }

    // Try to parse the JSON
    let data;
    try {
      data = JSON.parse(rawBody);
    } catch (parseError) {
      console.error('JSON parse error:', parseError);
      return NextResponse.json(
        { error: 'Invalid JSON in request body' },
        { status: 400 }
      );
    }
    
    // Validate the incoming data structure
    if (!data || typeof data !== 'object') {
      return NextResponse.json(
        { error: 'Request body must be a JSON object' },
        { status: 400 }
      );
    }

    // Validate required fields
    if (!data.source_code) {
      return NextResponse.json(
        { error: 'source_code is required' },
        { status: 400 }
      );
    }

    // Reset files at the start of new agent processing session
    const agentCodeFilePath = path.join(process.cwd(), 'processed_agent_code.json');
    const logsFilePath = path.join(process.cwd(), 'processed_logs.json');
    
    // Check if this is the first agent in a new session
    let shouldReset = false;
    
    try {
      // Check if files exist and if they should be reset
      const existingContent = await readFile(agentCodeFilePath, 'utf-8');
      const existingData = JSON.parse(existingContent);
      
      // Reset if the incoming agent has a different timestamp prefix (new workflow)
      if (Array.isArray(existingData) && existingData.length > 0) {
        const lastTimestamp = existingData[existingData.length - 1]?.timestamp || '';
        const newTimestamp = data.timestamp || '';
        
        // Extract workflow ID (first 15 characters: "20250523_185214")
        const lastWorkflowId = lastTimestamp.slice(0, 15);
        const newWorkflowId = newTimestamp.slice(0, 15);
        
        // Reset if this is a new workflow
        shouldReset = lastWorkflowId !== newWorkflowId;
      }
    } catch (error) {
      // Files don't exist, no need to reset
      shouldReset = false;
    }

    // Reset both files if this is a new workflow session
    if (shouldReset) {
      console.log('🔄 Resetting processed files for new workflow session');
      await writeFile(agentCodeFilePath, JSON.stringify([], null, 2));
      await writeFile(logsFilePath, JSON.stringify([], null, 2));
    }

    // Analyze vulnerabilities in the source code
    const sourceCode = data.source_code;
    const vulnerabilities = await analyzeVulnerabilities(sourceCode);

    // Process the agent code data
    const processedEntry: AgentCodeEntry = {
      agent_id: data.agent_id || 'unknown',
      agent_name: data.agent_name || 'Unknown Agent',
      source_code: sourceCode,
      description: data.description || '',
      file_path: data.file_path || '',
      dependencies: Array.isArray(data.dependencies) ? data.dependencies : [],
      timestamp: data.timestamp || new Date().toISOString(),
      processed_at: new Date().toISOString(),
      vulnerabilities: vulnerabilities
    };

    let existingData: AgentCodeEntry[] = [];
    
    // Try to read existing data
    try {
      const existingContent = await readFile(agentCodeFilePath, 'utf-8');
      existingData = JSON.parse(existingContent);
      if (!Array.isArray(existingData)) {
        existingData = [];
      }
    } catch (error) {
      // File doesn't exist or is invalid, start with empty array
      existingData = [];
    }
    
    // Add the new entry
    existingData.push(processedEntry);
    
    // Keep only the last 100 entries to prevent file from growing too large
    if (existingData.length > 100) {
      existingData = existingData.slice(-100);
    }
    
    // Write the updated data back to the file
    await writeFile(agentCodeFilePath, JSON.stringify(existingData, null, 2));
    
    console.log(`Processed agent code entry: ${processedEntry.agent_name} (${processedEntry.agent_id}) - ${vulnerabilities.length} vulnerabilities found`);
    
    return NextResponse.json({ 
      success: true, 
      message: 'Agent code processed successfully',
      processed_entry: processedEntry
    });
    
  } catch (error) {
    console.error('Error processing agent code:', error);
    return NextResponse.json(
      { 
        error: 'Failed to process agent code',
        details: error instanceof Error ? error.message : 'Unknown error'
      }, 
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({ 
    message: 'Use POST to submit agent code data for processing' 
  });
}

export async function DELETE() {
  try {
    // Reset both processed_agent_code.json and processed_logs.json
    // Using synchronous fs.writeFileSync for simplicity in DELETE, can be async too.
    fs.writeFileSync(agentCodeFilePath, JSON.stringify([], null, 2));
    fs.writeFileSync(logsFilePath, JSON.stringify([], null, 2)); 
    console.log('🗑️ Cleared processed_agent_code.json and processed_logs.json');
    return NextResponse.json({ message: 'Processed agent code and logs cleared' });
  } catch (error) {
    console.error('Failed to clear processed files:', error);
    return NextResponse.json({ error: 'Failed to clear data' }, { status: 500 });
  }
} 