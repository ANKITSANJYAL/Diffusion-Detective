import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  // State management
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(50)
  const [results, setResults] = useState(null)
  const [logs, setLogs] = useState([])
  const [displayedText, setDisplayedText] = useState('')
  const [error, setError] = useState(null)
  const [topTokens, setTopTokens] = useState([]) // Track current attention focus
  
  // Form state
  const [prompt, setPrompt] = useState('A majestic lion standing on a mountain peak at sunset')
  const [numSteps, setNumSteps] = useState(50)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [interventionActive, setInterventionActive] = useState(true)
  const [interventionStrength, setInterventionStrength] = useState(1.0)
  const [interventionStart, setInterventionStart] = useState(40)
  const [interventionEnd, setInterventionEnd] = useState(20)
  
  const terminalRef = useRef(null)
  const textToType = useRef('')
  const typingIndex = useRef(0)

  // Typewriter effect for terminal
  useEffect(() => {
    if (!textToType.current || typingIndex.current >= textToType.current.length) {
      return
    }

    const timer = setTimeout(() => {
      setDisplayedText(textToType.current.substring(0, typingIndex.current + 1))
      typingIndex.current += 1
    }, 10) // Fast typing speed

    return () => clearTimeout(timer)
  }, [displayedText, textToType.current])

  // Group consecutive similar logs to avoid clutter
  const groupLogs = (logsList) => {
    if (!Array.isArray(logsList) || logsList.length === 0) return []
    
    const grouped = []
    let currentGroup = null
    
    logsList.forEach((log, idx) => {
      // Check if this is a structured log step
      const isStepLog = typeof log === 'string' && log.match(/^\[Step \d+\]/)
      
      if (!isStepLog || !currentGroup) {
        // Start new group or add non-step log
        grouped.push(log)
        currentGroup = isStepLog ? { phase: log.match(/\] (.*?):/)?.[1], count: 1, lastIdx: idx } : null
      } else {
        // Check if same phase as current group
        const phase = log.match(/\] (.*?):/)?.[1]
        if (phase === currentGroup.phase && idx - currentGroup.lastIdx === 1) {
          currentGroup.count++
          currentGroup.lastIdx = idx
          // Replace last entry with grouped message
          if (currentGroup.count === 2) {
            grouped[grouped.length - 1] = `[Steps ${logsList[idx-1].match(/Step (\d+)/)[1]}-...] ${phase}: Processing...`
          } else {
            const startStep = logsList[idx - currentGroup.count + 1].match(/Step (\d+)/)[1]
            const endStep = log.match(/Step (\d+)/)[1]
            grouped[grouped.length - 1] = `[Steps ${startStep}-${endStep}] ${phase}: Processing ${currentGroup.count} steps...`
          }
        } else {
          grouped.push(log)
          currentGroup = { phase, count: 1, lastIdx: idx }
        }
      }
    })
    
    return grouped
  }

  // Highlight keywords in text
  const highlightKeywords = (text) => {
    const keywords = ['Color', 'Shape', 'Texture', 'Intervention', 'Attention', 'Focus', 'Natural', 'Controlled', 'Phase', 'Structure', 'Confidence']
    let highlighted = text
    
    keywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi')
      highlighted = highlighted.replace(regex, `<span class="text-cyan-400 font-bold neon-glow">${keyword}</span>`)
    })
    
    return highlighted
  }

  const handleGenerate = async () => {
    setIsGenerating(true)
    setCurrentStep(0)
    setLogs([])
    setError(null)
    setResults(null)
    setDisplayedText('')
    textToType.current = ''
    typingIndex.current = 0
    setTotalSteps(numSteps)

    try {
      // Build request payload
      const params = {
        prompt,
        num_inference_steps: numSteps,
        guidance_scale: guidanceScale,
        intervention_active: interventionActive,
        intervention_strength: interventionStrength,
        intervention_step_start: interventionStart,
        intervention_step_end: interventionEnd
      }
      
      // Add initial log
      setLogs(prev => [...prev, '🔍 Initializing Diffusion Detective...'])
      setLogs(prev => [...prev, `📝 Prompt: "${params.prompt}"`])
      setLogs(prev => [...prev, `⚙️ Steps: ${params.num_inference_steps} | Guidance: ${params.guidance_scale}`])
      
      if (params.intervention_active) {
        setLogs(prev => [...prev, `🧪 Intervention ACTIVE | Strength: ${params.intervention_strength}`])
        setLogs(prev => [...prev, `🎯 Intervention Zone: Steps ${params.intervention_step_end}-${params.intervention_step_start}`])
      } else {
        setLogs(prev => [...prev, '🔬 Natural generation (no intervention)'])
      }
      
      setLogs(prev => [...prev, ''])
      setLogs(prev => [...prev, '⏳ Generating images...'])
      setLogs(prev => [...prev, ''])

      // Simulate step progress for timeline animation
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + 1
          if (next >= numSteps) {
            clearInterval(stepInterval)
            return numSteps
          }
          
          // No fake logs - wait for real backend logs
          return next
        })
      }, 100) // Fast step animation

      const response = await axios.post(`${API_BASE_URL}/generate`, params, {
        timeout: 300000 // 5 minute timeout
      })

      clearInterval(stepInterval)
      setCurrentStep(numSteps)

      if (response.data.success) {
        console.log('Generation successful! Response:', {
          hasBaselineImage: !!response.data.image_baseline,
          hasIntervenedImage: !!response.data.image_intervened,
          baselineImagePrefix: response.data.image_baseline?.substring(0, 50),
          intervenedImagePrefix: response.data.image_intervened?.substring(0, 50),
          logsCount: response.data.reasoning_logs?.length,
          logsStructured: Array.isArray(response.data.reasoning_logs) && typeof response.data.reasoning_logs[0] === 'object'
        })
        
        // Process structured reasoning logs (now grouped)
        if (response.data.reasoning_logs && response.data.reasoning_logs.length > 0) {
          response.data.reasoning_logs.forEach(log => {
            if (typeof log === 'object') {
              // Check if this is a grouped log
              if (log.grouped && log.step_range) {
                const prefix = log.intervention_active ? '[INJECTION] ' : ''
                const logText = `[Steps ${log.step_range}] ${log.phase}: ${prefix}${log.message}`
                setLogs(prev => [...prev, logText])
              } else {
                // Regular structured log format
                const prefix = log.intervention_active ? '[INJECTION] ' : ''
                const logText = `[Step ${log.step}] ${log.phase}: ${prefix}${log.message}`
                setLogs(prev => [...prev, logText])
              }
              
              // Update top tokens from metadata
              if (log.metadata && log.metadata.top_tokens && log.metadata.top_tokens.length > 0) {
                setTopTokens(log.metadata.top_tokens)
              }
            } else {
              // Fallback for string logs
              setLogs(prev => [...prev, log])
            }
          })
        }
        
        setLogs(prev => [...prev, ''])
        setLogs(prev => [...prev, '🕵️ DETECTIVE\'S NARRATIVE:'])
        setLogs(prev => [...prev, '━━━━━━━━━━━━━━━━━━━━━━━━━━━━'])
        
        // Set results which will trigger the image display
        setResults(response.data)
        
        // Start typewriter effect for narrative
        textToType.current = response.data.narrative_text
        typingIndex.current = 0
        setDisplayedText('')
      }

    } catch (err) {
      console.error('Generation error:', err)
      setError(err.response?.data?.detail || err.message || 'Unknown error occurred')
      setLogs(prev => [...prev, ''])
      setLogs(prev => [...prev, `❌ ERROR: ${err.response?.data?.detail || err.message}`])
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="min-h-screen bg-cyber-black text-neon-green flex">
      
      {/* ========== SIDEBAR (LEFT) ========== */}
      <motion.aside
        className="w-80 bg-cyber-black border-r-2 border-neon-green/30 p-6 flex flex-col"
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        {/* Logo */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold neon-glow mb-2">
            🔍 DIFFUSION DETECTIVE
          </h1>
          <p className="text-xs text-neon-green/60">
            Interpretable & Intervene-able AI
          </p>
        </div>

        {/* Input Controls */}
        <div className="space-y-4 flex-1 overflow-y-auto">
          
          {/* Prompt */}
          <div>
            <label className="block text-xs font-bold mb-2 text-neon-green/80">
              📝 PROMPT
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-cyber-black border border-neon-green/50 rounded px-3 py-2 text-sm text-neon-green focus:border-neon-green focus:outline-none font-mono"
              rows="3"
              placeholder="Describe your vision..."
            />
          </div>

          {/* Num Steps */}
          <div>
            <label className="block text-xs font-bold mb-2 text-neon-green/80">
              ⚙️ INFERENCE STEPS: {numSteps}
            </label>
            <input
              type="range"
              min="20"
              max="100"
              value={numSteps}
              onChange={(e) => setNumSteps(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Guidance Scale */}
          <div>
            <label className="block text-xs font-bold mb-2 text-neon-green/80">
              🎯 GUIDANCE SCALE: {guidanceScale}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              step="0.5"
              value={guidanceScale}
              onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Intervention Toggle */}
          <div className="border-t border-neon-green/30 pt-4">
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={interventionActive}
                onChange={(e) => setInterventionActive(e.target.checked)}
                className="w-5 h-5"
              />
              <span className="text-sm font-bold text-neon-green">
                🧪 ENABLE INTERVENTION
              </span>
            </label>
          </div>

          {/* Intervention Controls */}
          {interventionActive && (
            <motion.div
              className="space-y-4 border border-neon-red/50 rounded p-3 bg-neon-red/5"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
            >
              <div>
                <label className="block text-xs font-bold mb-2 text-neon-red">
                  💉 INTERVENTION STRENGTH: {interventionStrength.toFixed(1)}x
                </label>
                <input
                  type="range"
                  min="0"
                  max="3"
                  step="0.1"
                  value={interventionStrength}
                  onChange={(e) => setInterventionStrength(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-xs font-bold mb-2 text-neon-red">
                  🎯 START STEP: {interventionStart}
                </label>
                <input
                  type="range"
                  min={interventionEnd}
                  max={numSteps}
                  value={interventionStart}
                  onChange={(e) => setInterventionStart(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-xs font-bold mb-2 text-neon-red">
                  🏁 END STEP: {interventionEnd}
                </label>
                <input
                  type="range"
                  min="1"
                  max={interventionStart}
                  value={interventionEnd}
                  onChange={(e) => setInterventionEnd(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            </motion.div>
          )}
        </div>

        {/* Run Button */}
        <motion.button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={`w-full py-4 mt-6 font-bold text-lg rounded-lg border-2 transition-all ${
            isGenerating
              ? 'bg-gray-800 border-gray-600 text-gray-500 cursor-not-allowed'
              : 'bg-neon-green/10 border-neon-green text-neon-green hover:bg-neon-green hover:text-cyber-black neon-glow'
          }`}
          whileHover={!isGenerating ? { scale: 1.05 } : {}}
          whileTap={!isGenerating ? { scale: 0.95 } : {}}
        >
          {isGenerating ? '⏳ ANALYZING...' : '🚀 RUN ANALYSIS'}
        </motion.button>

        {/* Footer */}
        <div className="mt-6 text-center text-xs text-neon-green/40">
          <p>Powered by Stable Diffusion</p>
          <p className="mt-1">MPS Acceleration Enabled</p>
        </div>
      </motion.aside>

      {/* ========== MAIN DASHBOARD (RIGHT) ========== */}
      <main className="flex-1 flex flex-col overflow-hidden">
        
        {/* Timeline Header */}
        <motion.div
          className="bg-cyber-black border-b-2 border-neon-green/30 p-4"
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {isGenerating || results ? (
            <div>
              <h3 className="text-xs font-bold text-neon-green/70 mb-2">DIFFUSION TIMELINE</h3>
              <div className="flex items-center space-x-2 relative">
                {/* Step markers */}
                {Array.from({ length: 10 }).map((_, i) => {
                  const step = Math.floor((numSteps / 10) * (10 - i))
                  const progress = (currentStep / numSteps) * 100
                  const stepProgress = ((10 - i) / 10) * 100
                  const isActive = progress >= stepProgress
                  const isIntervention = interventionActive && step >= interventionEnd && step <= interventionStart
                  
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center">
                      <motion.div
                        className={`w-full h-2 rounded-full ${
                          isIntervention && isActive
                            ? 'bg-neon-red animate-pulse'
                            : isActive
                            ? 'bg-neon-green'
                            : 'bg-gray-700'
                        }`}
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: isActive ? 1 : 0 }}
                        transition={{ duration: 0.3 }}
                      />
                      <span className={`text-xs mt-1 ${isActive ? 'text-neon-green' : 'text-gray-600'}`}>
                        {step}
                      </span>
                      {isIntervention && isActive && (
                        <span className="text-xs text-neon-red font-bold mt-1">💉</span>
                      )}
                    </div>
                  )
                })}
                
                {/* Animated Scan Line */}
                {isGenerating && (
                  <motion.div
                    className="absolute top-0 w-0.5 h-8 bg-white shadow-lg shadow-white/70 z-10"
                    style={{ left: 0 }}
                    animate={{ left: ['0%', '100%'] }}
                    transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-2">
              <h3 className="text-sm text-neon-green/50">⚡ Ready for Analysis</h3>
            </div>
          )}
        </motion.div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden p-6">
          
          <AnimatePresence mode="wait">
            {/* Awaiting State */}
            {!isGenerating && !results && (
              <motion.div
                key="awaiting"
                className="flex-1 flex items-center justify-center"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
              >
                <div className="text-center">
                  <motion.div
                    className="text-8xl mb-4"
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  >
                    🔍
                  </motion.div>
                  <h2 className="text-2xl font-bold text-neon-green/70 mb-2">
                    AWAITING MISSION PARAMETERS
                  </h2>
                  <p className="text-sm text-neon-green/50">
                    Configure your analysis settings and hit RUN ANALYSIS
                  </p>
                </div>
              </motion.div>
            )}

            {/* Generating State */}
            {isGenerating && (
              <motion.div
                key="generating"
                className="flex-1 flex items-center justify-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="text-center">
                  <motion.div
                    className="text-8xl mb-4"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  >
                    🔬
                  </motion.div>
                  <h2 className="text-2xl font-bold text-neon-green neon-glow mb-2">
                    ANALYSIS IN PROGRESS
                  </h2>
                  <p className="text-sm text-neon-green/70">
                    Step {currentStep} / {numSteps}
                  </p>
                  <div className="mt-4 w-64 mx-auto">
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-neon-green to-neon-blue"
                        initial={{ width: '0%' }}
                        animate={{ width: `${(currentStep / numSteps) * 100}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Results State */}
            {results && !isGenerating && (
              <motion.div
                key="results"
                className="flex-1 flex flex-col overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
              >
                {/* Comparison Header */}
                <div className="mb-4">
                  <h3 className="text-lg font-bold text-neon-green neon-glow">
                    🔬 COMPARATIVE ANALYSIS
                  </h3>
                  <div className="flex justify-between text-xs text-neon-green/70 mt-2">
                    <span className="flex items-center">
                      <span className="w-2 h-2 bg-neon-blue rounded-full mr-2" />
                      Natural (Baseline)
                    </span>
                    <span className="flex items-center">
                      <span className="w-2 h-2 bg-neon-purple rounded-full mr-2" />
                      Controlled (Intervened)
                    </span>
                  </div>
                </div>

                {/* Comparison Slider */}
                <div className="relative rounded-lg overflow-hidden border-2 border-neon-green/50 bg-black">
                  {results.image_baseline && results.image_intervened ? (
                    <div className="grid grid-cols-2 gap-2">
                      {/* Baseline Image */}
                      <div className="relative">
                        <img
                          src={results.image_baseline.startsWith('data:') ? results.image_baseline : `data:image/png;base64,${results.image_baseline}`}
                          alt="Baseline"
                          className="w-full h-auto"
                        />
                        <div className="absolute top-2 left-2 bg-cyber-black/90 border border-neon-green px-3 py-1 rounded text-xs text-neon-green font-bold shadow-lg">
                          BASELINE (Natural)
                        </div>
                      </div>
                      
                      {/* Intervened Image */}
                      <div className="relative">
                        <img
                          src={results.image_intervened.startsWith('data:') ? results.image_intervened : `data:image/png;base64,${results.image_intervened}`}
                          alt="Intervened"
                          className="w-full h-auto"
                        />
                        <div className="absolute top-2 left-2 px-3 py-1 rounded text-xs font-bold shadow-lg" style={{ backgroundColor: '#FF0055', border: '1px solid #FF0055', color: 'white' }}>
                          INTERVENTION (Forced)
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-96 flex items-center justify-center text-neon-green/50">
                      <p>Loading images...</p>
                    </div>
                  )}
                </div>

                {/* Download Buttons */}
                <div className="mt-4 flex gap-3 justify-center">
                  <a
                    href={results.image_baseline}
                    download="baseline_image.png"
                    className="cyber-button text-xs"
                  >
                    ⬇️ Download Baseline
                  </a>
                  <a
                    href={results.image_intervened}
                    download="intervened_image.png"
                    className="cyber-button text-xs"
                  >
                    ⬇️ Download Intervened
                  </a>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Terminal Log at Bottom */}
        <motion.div
          className="h-64 border-t-2 border-neon-green/30 bg-cyber-black p-4 overflow-y-auto font-mono text-xs"
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          ref={terminalRef}
        >
          <div className="mb-2 border-b border-neon-green/30 pb-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-neon-green">
                🕵️ DETECTIVE'S LOG
              </h3>
              <span className={`text-xs ${isGenerating ? 'text-neon-green animate-pulse' : 'text-gray-600'}`}>
                {isGenerating ? '● ACTIVE' : '○ IDLE'}
              </span>
            </div>
            
            {/* Top Tokens Display */}
            {topTokens.length > 0 && (
              <div className="mt-2 pt-2 border-t border-neon-cyan/20">
                <div className="text-xs text-neon-cyan/70 mb-1">🎯 Attention Focus:</div>
                <div className="flex gap-2 flex-wrap">
                  {topTokens.map((token, idx) => (
                    <div key={idx} className="bg-cyber-dark border border-neon-cyan/30 px-2 py-0.5 rounded text-xs">
                      <span className="text-neon-cyan font-bold">{token.token.toUpperCase()}</span>
                      <span className="text-neon-green/70 ml-1">{token.confidence}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          <div className="space-y-1">
            {logs.map((log, idx) => {
              const isInjection = log.includes('[INJECTION]')
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={isInjection ? 'text-neon-red font-bold' : 'text-neon-green/90'}
                  dangerouslySetInnerHTML={{ __html: highlightKeywords(log) }}
                />
              )
            })}
            
            {/* Typewriter Effect for narrative */}
            {displayedText && (
              <motion.div
                className="mt-4 pt-4 border-t border-neon-green/30"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div 
                  className="text-neon-green/90 whitespace-pre-wrap"
                  dangerouslySetInnerHTML={{ __html: highlightKeywords(displayedText) }}
                />
                <span className="inline-block w-2 h-4 bg-neon-green ml-1 animate-pulse" />
              </motion.div>
            )}
          </div>
        </motion.div>

      </main>
    </div>
  )
}

export default App
