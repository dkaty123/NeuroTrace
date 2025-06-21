"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Settings, Shield, Bell, Palette, Database, Key, User, Monitor } from "lucide-react";
import Sidebar from "../components/Sidebar";

export default function SettingsPage() {
  return (
    <div className="flex h-screen bg-black">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
          <p className="text-zinc-400">Configure your NeuroTrace security platform</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-zinc-900/50 border-zinc-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Shield className="w-5 h-5 text-cyan-400" />
                Security Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Real-time Threat Detection</span>
                <Badge className="bg-green-500/20 text-green-400">Enabled</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Auto-remediation</span>
                <Badge className="bg-orange-500/20 text-orange-400">Disabled</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Alert Threshold</span>
                <span className="text-zinc-400">High</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/50 border-zinc-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Bell className="w-5 h-5 text-indigo-400" />
                Notifications
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Critical Alerts</span>
                <Badge className="bg-red-500/20 text-red-400">Enabled</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Email Reports</span>
                <Badge className="bg-blue-500/20 text-blue-400">Daily</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Slack Integration</span>
                <Badge className="bg-zinc-500/20 text-zinc-400">Pending</Badge>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/50 border-zinc-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <Database className="w-5 h-5 text-purple-400" />
                Data Management
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Retention Policy</span>
                <span className="text-zinc-400">90 days</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Backup Frequency</span>
                <span className="text-zinc-400">Daily</span>
              </div>
              <Button size="sm" variant="outline" className="w-full">
                Configure Data Export
              </Button>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/50 border-zinc-700">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-white">
                <User className="w-5 h-5 text-teal-400" />
                Profile Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Two-Factor Auth</span>
                <Badge className="bg-green-500/20 text-green-400">Enabled</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-300">Session Timeout</span>
                <span className="text-zinc-400">4 hours</span>
              </div>
              <Button size="sm" variant="outline" className="w-full">
                Change Password
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="mt-8 text-center">
          <Button className="bg-cyan-500 hover:bg-cyan-600">
            Save All Settings
          </Button>
        </div>
        </div>
      </main>
    </div>
  );
} 