import { NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import path from 'path';

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), 'processed_agent_code.json');
    
    // Try to read the processed agent code file
    try {
      const fileContent = await readFile(filePath, 'utf-8');
      const data = JSON.parse(fileContent);
      
      // Validate it's an array
      if (!Array.isArray(data)) {
        return NextResponse.json([], { status: 200 });
      }
      
      return NextResponse.json(data, { 
        status: 200,
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
          'Pragma': 'no-cache',
          'Expires': '0'
        }
      });
      
    } catch (fileError) {
      // File doesn't exist or is invalid, return empty array
      return NextResponse.json([], { status: 200 });
    }
    
  } catch (error) {
    console.error('Error reading agent code data:', error);
    return NextResponse.json(
      { error: 'Failed to read agent code data' }, 
      { status: 500 }
    );
  }
} 