import { NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import fs from 'fs';
import path from 'path';

export async function POST() {
  try {
    const agentCodeFilePath = path.join(process.cwd(), 'processed_agent_code.json');
    const logsFilePath = path.join(process.cwd(), 'processed_logs.json');
    
    // Reset both files to empty arrays
    await Promise.all([
      writeFile(agentCodeFilePath, JSON.stringify([], null, 2)),
      writeFile(logsFilePath, JSON.stringify([], null, 2))
    ]);
    
    console.log('🔄 Reset processed data files - server ready for new agents');
    
    return NextResponse.json({ 
      success: true,
      message: 'All processed data has been reset successfully',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error resetting processed data:', error);
    return NextResponse.json(
      { error: 'Failed to reset processed data' }, 
      { status: 500 }
    );
  }
}

export async function DELETE() {
  try {
    const agentCodeFilePath = path.join(process.cwd(), 'processed_agent_code.json');
    const logsFilePath = path.join(process.cwd(), 'processed_logs.json');
    
    // Delete the files entirely
    const deletePromises = [];
    
    if (fs.existsSync(agentCodeFilePath)) {
      deletePromises.push(unlink(agentCodeFilePath));
    }
    
    if (fs.existsSync(logsFilePath)) {
      deletePromises.push(unlink(logsFilePath));
    }
    
    await Promise.all(deletePromises);
    
    console.log('🗑️ Deleted processed data files - clean slate for new session');
    
    return NextResponse.json({ 
      success: true,
      message: 'All processed data files have been deleted',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error deleting processed data files:', error);
    return NextResponse.json(
      { error: 'Failed to delete processed data files' }, 
      { status: 500 }
    );
  }
} 