import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const filePath = path.join(process.cwd(), 'processed_logs.json');
    
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      return NextResponse.json([], { status: 200 }); // Return empty array if file doesn't exist
    }
    
    // Read and parse the file
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(fileContent);
    
    return NextResponse.json(data, { 
      status: 200,
      headers: {
        'Cache-Control': 'no-store, max-age=0'
      }
    });
  } catch (error) {
    console.error('Error reading processed logs:', error);
    return NextResponse.json([], { status: 200 }); // Return empty array on error
  }
} 