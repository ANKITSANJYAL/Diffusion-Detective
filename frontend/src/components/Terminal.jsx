import { forwardRef, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const Terminal = forwardRef(({ logs, isActive }, ref) => {
  const terminalEndRef = useRef(null)

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  const highlightKeywords = (text) => {
    const keywords = {
      'Focus': 'text-neon-blue',
      'Shape': 'text-neon-purple',
      'Color': 'text-neon-blue',
      'Texture': 'text-neon-purple',
      'Intervention': 'text-neon-red',
      'Steering': 'text-neon-red',
      'ERROR': 'text-neon-red font-bold',
      'complete': 'text-neon-green font-bold',
      'ACTIVE': 'text-neon-red font-bold'
    }

    let highlightedText = text
    
    Object.entries(keywords).forEach(([keyword, className]) => {
      const regex = new RegExp(`(${keyword})`, 'gi')
      highlightedText = highlightedText.replace(
        regex,
        `<span class="${className}">$1</span>`
      )
    })

    return highlightedText
  }

  return (
    <div className="terminal-window scanline relative" ref={ref}>
      
      {/* Header */}
      <div className="flex items-center justify-between mb-3 border-b border-neon-green/30 pb-2">
        <h3 className="text-sm font-bold text-neon-green/90">
          💻 INVESTIGATION LOG
        </h3>
        <div className="flex items-center space-x-2 text-xs">
          <span className={`${isActive ? 'text-neon-green animate-pulse' : 'text-neon-green/50'}`}>
            {isActive ? '● ACTIVE' : '○ IDLE'}
          </span>
        </div>
      </div>

      {/* Terminal Content */}
      <div className="bg-cyber-black/80 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
        
        {/* Welcome Message */}
        {logs.length === 0 && (
          <div className="text-neon-green/50">
            <p>╔════════════════════════════════════════╗</p>
            <p>║   DIFFUSION DETECTIVE v1.0.0          ║</p>
            <p>║   Awaiting Analysis Request...         ║</p>
            <p>╚════════════════════════════════════════╝</p>
            <p className="mt-2">&gt; Ready to investigate_</p>
          </div>
        )}

        {/* Log Entries */}
        <AnimatePresence>
          {logs.map((log, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2, delay: index * 0.02 }}
              className="mb-1"
            >
              <span 
                dangerouslySetInnerHTML={{ __html: highlightKeywords(log) }}
                className="text-neon-green/90"
              />
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Cursor */}
        {isActive && (
          <motion.span
            className="inline-block w-2 h-4 bg-neon-green ml-1"
            animate={{ opacity: [1, 0, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}

        {/* Auto-scroll anchor */}
        <div ref={terminalEndRef} />
      </div>

      {/* Stats Bar */}
      <div className="mt-2 flex items-center justify-between text-xs text-neon-green/50">
        <span>Lines: {logs.length}</span>
        <span>Status: {isActive ? 'Processing' : 'Ready'}</span>
      </div>
    </div>
  )
})

Terminal.displayName = 'Terminal'

export default Terminal
