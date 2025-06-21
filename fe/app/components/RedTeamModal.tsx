import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { RedTeamResult } from '../lib/types/redTeam';
import { RedTeamService } from '../lib/services/redTeamService';
import { Loader2, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react';

interface RedTeamModalProps {
  isOpen: boolean;
  onClose: () => void;
  onTestComplete: (results: RedTeamResult[]) => void;
}

export function RedTeamModal({ isOpen, onClose, onTestComplete }: RedTeamModalProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [currentResults, setCurrentResults] = useState<RedTeamResult[]>([]);
  const redTeamService = RedTeamService.getInstance();

  const runTests = async () => {
    setIsRunning(true);
    try {
      const results = await redTeamService.runAllTests();
      setCurrentResults(results);
      onTestComplete(results);
    } catch (error) {
      console.error('Error running red team tests:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const getStatusColor = (passed: boolean) => {
    return passed
      ? "text-green-400 bg-green-500/10 border-green-500/20"
      : "text-red-400 bg-red-500/10 border-red-500/20";
  };

  const getStatusIcon = (passed: boolean) => {
    return passed ? (
      <CheckCircle2 className="w-4 h-4" />
    ) : (
      <XCircle className="w-4 h-4" />
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-2xl bg-zinc-900 border-zinc-800">
        <DialogHeader>
          <DialogTitle className="text-xl font-bold text-white">Red Team Testing</DialogTitle>
          <DialogDescription className="text-zinc-400">
            Run adversarial prompts to test system safety and alignment
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {!isRunning && currentResults.length === 0 && (
            <div className="text-center py-8">
              <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-zinc-300 mb-2">
                Ready to Run Tests
              </h3>
              <p className="text-zinc-500 mb-4">
                This will run a series of adversarial prompts to test system safety
              </p>
              <Button
                onClick={runTests}
                className="bg-purple-600 hover:bg-purple-700 text-white"
              >
                Start Red Team Testing
              </Button>
            </div>
          )}

          {isRunning && (
            <div className="text-center py-8">
              <Loader2 className="w-12 h-12 text-purple-400 mx-auto mb-4 animate-spin" />
              <h3 className="text-lg font-medium text-zinc-300">
                Running Tests...
              </h3>
            </div>
          )}

          {!isRunning && currentResults.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-medium text-zinc-300">Results</h3>
                  <p className="text-sm text-zinc-500">
                    {currentResults.filter(r => r.passed).length} passed,{' '}
                    {currentResults.filter(r => !r.passed).length} failed
                  </p>
                </div>
                <Button
                  onClick={runTests}
                  className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300"
                >
                  Run Again
                </Button>
              </div>

              {currentResults.map((result) => (
                <div
                  key={result.id}
                  className="border border-zinc-800 rounded-lg p-4 space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(result.passed)}
                      <span className="font-medium text-zinc-300">
                        Test {result.prompt_id}
                      </span>
                    </div>
                    <Badge className={getStatusColor(result.passed)}>
                      {result.passed ? "Passed" : "Failed"}
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-zinc-500 mb-1">Input</p>
                      <p className="text-zinc-300 font-mono bg-black/30 p-2 rounded">
                        {result.input}
                      </p>
                    </div>
                    <div>
                      <p className="text-zinc-500 mb-1">Output</p>
                      <p className="text-zinc-300 font-mono bg-black/30 p-2 rounded">
                        {result.output}
                      </p>
                    </div>
                  </div>

                  {result.metadata && (
                    <div className="flex gap-4 text-xs text-zinc-500">
                      <span>
                        Time: {Math.round(result.metadata.execution_time ?? 0)}ms
                      </span>
                      <span>Tokens: {result.metadata.tokens_used}</span>
                      <span>Model: {result.metadata.model_version}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
} 