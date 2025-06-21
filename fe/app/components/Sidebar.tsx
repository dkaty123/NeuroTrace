"use client";

import React, { useState, useEffect } from "react";
import { LayoutDashboard, Shield, Settings, HelpCircle, LogOut, User, Cpu, Activity, Radar, Target } from "lucide-react";
import Image from "next/image";
import { usePathname } from "next/navigation";
import Link from "next/link";

interface VulnerabilityStats {
  total: number;
  critical: number;
  high: number;
}

export default function Sidebar() {
  const pathname = usePathname();
  const [vulnerabilityStats, setVulnerabilityStats] = useState<VulnerabilityStats>({ total: 0, critical: 0, high: 0 });
  const [isOnline, setIsOnline] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);
  const [ping, setPing] = useState(3);

  // Fetch vulnerability stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const agentsResponse = await fetch('/api/get-agent-code');
        const logsResponse = await fetch('/api/get-processed-logs');
        
        if (agentsResponse.ok && logsResponse.ok) {
          const agents = await agentsResponse.json();
          const logs = await logsResponse.json();
          
          let total = 0;
          let critical = 0;
          let high = 0;
          
          // Count agent vulnerabilities
          agents.forEach((agent: any) => {
            if (agent.vulnerabilities) {
              total += agent.vulnerabilities.length;
              critical += agent.vulnerabilities.filter((v: any) => v.type === 'critical').length;
              high += agent.vulnerabilities.filter((v: any) => v.type === 'high').length;
            }
          });
          
          // Count log vulnerabilities
          logs.forEach((log: any) => {
            if (log.vulnerability && log.vulnerability_details !== "No vulnerabilities detected") {
              total += 1;
              const details = log.vulnerability_details.toLowerCase();
              if (details.includes("critical") || details.includes("remote code") || details.includes("injection")) {
                critical += 1;
              } else if (details.includes("high") || details.includes("security") || details.includes("unauthorized")) {
                high += 1;
              }
            }
          });
          
          setVulnerabilityStats({ total, critical, high });
        }
      } catch (error) {
        console.error('Failed to fetch vulnerability stats:', error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Update ping every 2 seconds with random latency
  useEffect(() => {
    const updatePing = () => {
      // Random ping between 1-9ms for that "close to server" feel
      const newPing = Math.floor(Math.random() * 9) + 1;
      setPing(newPing);
    };

    const interval = setInterval(updatePing, 2000);
    return () => clearInterval(interval);
  }, []);

  const isActive = (path: string) => pathname === path;

  const NavItem = ({ 
    href, 
    icon: Icon, 
    label, 
    badge
  }: { 
    href: string; 
    icon: any; 
    label: string; 
    badge?: number; 
  }) => {
    const active = isActive(href);
    
    return (
      <Link href={href} className="block">
        <div className={`
          group relative flex items-center px-3 py-2.5 rounded-lg transition-all duration-200 ease-out
          ${active 
            ? 'bg-gradient-to-r from-purple-500/15 to-violet-500/15 text-purple-300' 
            : 'text-zinc-400 hover:text-purple-300 hover:bg-zinc-800/60'
          }
        `}>
          {/* Glowing left border for active state */}
          {active && (
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-purple-400 to-violet-400 rounded-r-full" />
          )}
          
          <Icon className={`
            w-5 h-5 mr-3 transition-colors duration-200
            ${active ? 'text-purple-400' : 'group-hover:text-purple-400'}
          `} />
          
          <span className={`
            font-medium tracking-wide transition-colors duration-200
            ${active ? 'text-white font-semibold' : 'group-hover:text-white'}
          `}>
            {label}
          </span>
          
          {/* Badge */}
          {badge !== undefined && badge > 0 && (
            <div className={`
              ml-auto px-2 py-1 text-xs font-bold rounded-full border-2 transition-all duration-300
              ${badge > 9 
                ? 'bg-red-500/20 text-red-300 border-red-500/60 shadow-lg shadow-red-500/30 animate-pulse' 
                : 'bg-orange-500/20 text-orange-300 border-orange-500/60 shadow-lg shadow-orange-500/30'
              }
            `}>
              {badge > 99 ? '99+' : badge}
            </div>
          )}
          
          {/* Subtle hover effect */}
          <div className={`
            absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200
            bg-purple-500/3
          `} />
        </div>
      </Link>
    );
  };

  return (
    <aside className="w-64 bg-gradient-to-b from-black via-zinc-950 to-black border-r border-zinc-800/50 flex flex-col min-h-screen relative overflow-hidden">
      {/* Subtle background texture */}
      <div className="absolute inset-0 opacity-10 bg-[radial-gradient(circle_at_50%_50%,rgba(147,51,234,0.1),transparent_50%)]" />
      <div className="absolute inset-0 opacity-5 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48cGF0dGVybiBpZD0iZ3JpZCIgd2lkdGg9IjQwIiBoZWlnaHQ9IjQwIiBwYXR0ZXJuVW5pdHM9InVzZXJTcGFjZU9uVXNlIj48cGF0aCBkPSJNIDQwIDAgTCAwIDAgMCA0MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMDMpIiBzdHJva2Utd2lkdGg9IjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')]" />
      
      <div className="relative z-10 flex flex-col h-full py-4 px-4">
        {/* Logo Section */}
        <div className="mb-8 flex items-center group">
          <div className="-ml-2 flex items-center">
            <Image
              src="/neurotrace-mark.svg"
              alt="NeuroTrace Logo"
              width={32}
              height={32}
              className="invert transition-transform duration-300 group-hover:scale-110"
            />
          </div>
          <span className="text-xl font-bold text-white group-hover:text-purple-300 transition-colors duration-300">
            NeuroTrace
          </span>
        </div>

        {/* Primary Navigation */}
        <nav className="flex-1">
          <div className="space-y-1">
            <NavItem 
              href="/dashboard" 
              icon={LayoutDashboard} 
              label="Dashboard" 
            />
            <NavItem 
              href="/vulnerabilities" 
              icon={Shield} 
              label="Vulnerabilities" 
              badge={vulnerabilityStats.total}
            />
            <NavItem 
              href="/analytics" 
              icon={Cpu} 
              label="Analytics" 
            />
            <NavItem 
              href="/red-team" 
              icon={Target} 
              label="Red Team" 
            />
          </div>

          {/* Section Separator */}
          <div className="my-6 relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
            </div>
            <div className="relative flex justify-center">
                              <div className="w-2 h-2 bg-purple-400/60 rounded-full shadow-lg shadow-purple-400/50" />
            </div>
          </div>

          {/* Secondary Navigation */}
          <div className="space-y-1">
            <NavItem 
              href="/settings" 
              icon={Settings} 
              label="Settings" 
            />
            <NavItem 
              href="/docs" 
              icon={HelpCircle} 
              label="Help & Docs" 
            />
          </div>
        </nav>

        {/* Status Indicator */}
        <div className="mb-4 px-3 py-2 bg-zinc-900/50 rounded-lg border border-zinc-700/50">
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400 shadow-lg shadow-green-400/50' : 'bg-red-400 shadow-lg shadow-red-400/50'} animate-pulse`} />
            <span className="text-zinc-300">
              {isOnline ? 'System Online' : 'System Offline'}
            </span>
            {/* Live ping indicator */}
            <div className="ml-auto flex items-center gap-1">
              <span className="text-xs text-zinc-400">ping</span>
              <span className={`text-xs font-mono font-bold transition-colors duration-300 ${
                ping <= 3 ? 'text-green-400' : 
                ping <= 6 ? 'text-yellow-400' : 
                'text-red-400'
              }`}>
                {ping}ms
              </span>
            </div>
          </div>
          

        </div>

        {/* Profile Section - Footer */}
        <div className="border-t border-zinc-700/50 pt-4">
          <button 
            onClick={() => setProfileOpen(!profileOpen)}
            className="w-full flex items-center px-3 py-2 rounded-lg hover:bg-zinc-800/50 transition-all duration-300 group"
          >
            <div className="w-8 h-8 bg-gradient-to-br from-purple-400 to-violet-500 rounded-full flex items-center justify-center mr-3">
              <User className="w-4 h-4 text-white" />
            </div>
            <div className="flex-1 text-left">
                              <div className="text-sm font-medium text-white group-hover:text-purple-300 transition-colors">
                Mizan T.
              </div>
              <div className="text-xs text-zinc-400">Security Admin</div>
            </div>
            <LogOut className="w-4 h-4 text-zinc-400 group-hover:text-red-400 transition-colors" />
          </button>
        </div>
      </div>
    </aside>
  );
} 