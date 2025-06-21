"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function HelpRedirect() {
  const router = useRouter()
  
  useEffect(() => {
    router.replace("/docs")
  }, [router])

  return (
    <div className="flex items-center justify-center min-h-screen bg-black">
      <div className="text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-4"></div>
        <p className="text-zinc-300">Redirecting to documentation...</p>
      </div>
    </div>
  )
} 