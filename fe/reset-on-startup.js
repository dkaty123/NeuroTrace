const fs = require('fs');
const path = require('path');

// This script runs on server startup to reset processed data files
// ensuring we start with a clean slate for new agent runs

async function resetProcessedData() {
  try {
    const agentCodeFilePath = path.join(process.cwd(), 'processed_agent_code.json');
    const logsFilePath = path.join(process.cwd(), 'processed_logs.json');
    
    // Reset both files to empty arrays
    const emptyArray = JSON.stringify([], null, 2);
    
    await Promise.all([
      fs.promises.writeFile(agentCodeFilePath, emptyArray),
      fs.promises.writeFile(logsFilePath, emptyArray)
    ]);
    
    console.log('🔄 Server startup: Reset processed data files - ready for new agents');
  } catch (error) {
    console.error('❌ Error resetting processed data on startup:', error);
  }
}

// Run the reset
resetProcessedData(); 