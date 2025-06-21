import type { Metadata } from "next";
import { Geist, Geist_Mono, Orbitron, Exo_2 } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const orbitron = Orbitron({
  variable: "--font-orbitron",
  subsets: ["latin"],
});

const exo2 = Exo_2({
  variable: "--font-exo2",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NeuroTrace - AI Agent Security Platform",
  description: "Real-time observability and security analysis for LangGraph AI agent workflows",
  keywords: ["AI", "LangGraph", "Observability", "Security", "Agents"],
  authors: [{ name: "NeuroTrace Team" }],
  openGraph: {
    title: "NeuroTrace - AI Agent Security Platform",
    description: "Real-time security analysis and vulnerability detection for Python AI agents and workflows.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "NeuroTrace - AI Agent Security Platform",
    description: "Real-time security analysis and vulnerability detection for Python AI agents and workflows.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${orbitron.variable} ${exo2.variable} antialiased`}
        suppressHydrationWarning={true}
      >
        {children}
      </body>
    </html>
  );
}
