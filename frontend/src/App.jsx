import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  // State management
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [totalSteps, setTotalSteps] = useState(50)
  const [currentStatus, setCurrentStatus] = useState('processing') // 'processing', 'decoding', 'analyzing', 'complete'
  const [results, setResults] = useState(null)
  const [logs, setLogs] = useState([])
  const [displayedText, setDisplayedText] = useState('')
  const [error, setError] = useState(null)
  const [topTokens, setTopTokens] = useState([]) // Track current attention focus
  
  // Form state
  const [prompt, setPrompt] = useState('A majestic tiger standing on a mountain peak at sunset')
  const [numSteps, setNumSteps] = useState(50)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [interventionActive, setInterventionActive] = useState(true)
  const [interventionStrength, setInterventionStrength] = useState(1.0)
  const [interventionStart, setInterventionStart] = useState(40)
  const [interventionEnd, setInterventionEnd] = useState(20)
  
  // v2.0: Semantic intervention controls
  const [targetConcept, setTargetConcept] = useState('')
  const [injectionAttribute, setInjectionAttribute] = useState('')
  const [autoDetectConcepts, setAutoDetectConcepts] = useState(true)
  const [detectedConcepts, setDetectedConcepts] = useState([])
  const [focusScores, setFocusScores] = useState({})
  
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

  // Process complete response from streaming
  const processCompleteResponse = (data) => {
    console.log('Generation successful! Response:', {
      hasBaselineImage: !!data.image_baseline,
      hasIntervenedImage: !!data.image_intervened,
      baselineImagePrefix: data.image_baseline?.substring(0, 50),
      intervenedImagePrefix: data.image_intervened?.substring(0, 50),
      logsCount: data.reasoning_logs?.length,
      logsStructured: Array.isArray(data.reasoning_logs) && typeof data.reasoning_logs[0] === 'object'
    })
    
    // Process structured reasoning logs (now LLM-generated or grouped)
    if (data.reasoning_logs && data.reasoning_logs.length > 0) {
      // Check if these are LLM-generated logs
      const isLLMGenerated = data.reasoning_logs[0]?.llm_generated === true
      
      if (isLLMGenerated) {
        // LLM-GENERATED LOGS - Display with special formatting including stats
        setLogs(prev => [...prev, ''])
        setLogs(prev => [...prev, 'NEURAL STATE ANALYSIS (LLM-Generated):'])
        setLogs(prev => [...prev, '━━━━━━━━━━━━━━━━━━━━━━━━━━━━'])
        setLogs(prev => [...prev, ''])
        
        data.reasoning_logs.forEach(log => {
          if (typeof log === 'object' && log.message) {
            // Create structured log entry with type for color coding
            const logEntry = {
              range: log.range || 'Unknown',
              type: log.type || 'normal',
              message: log.message,
              stats: log.stats || {},
              isStructured: true
            }
            setLogs(prev => [...prev, logEntry])
          }
        })
      } else {
        // Original grouped/structured logs
        data.reasoning_logs.forEach(log => {
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
            
            // v2.0: Update focus scores for multi-concept tracking
            if (log.metadata && log.metadata.focus) {
              setFocusScores(log.metadata.focus)
            }
          } else {
            // Fallback for string logs
            setLogs(prev => [...prev, log])
          }
        })
      }
    }
    
    setLogs(prev => [...prev, ''])
    setLogs(prev => [...prev, 'DETECTIVE\'S NARRATIVE:'])
    setLogs(prev => [...prev, '━━━━━━━━━━━━━━━━━━━━━━━━━━━━'])
    
    // Set results which will trigger the image display
    setResults(data)
    setCurrentStatus('complete')
    
    // Start typewriter effect for narrative
    textToType.current = data.narrative_text
    typingIndex.current = 0
    setDisplayedText('')
  }

  const handleGenerate = async () => {
    setIsGenerating(true)
    setCurrentStep(0)
    setCurrentStatus('processing')
    setLogs([])
    setError(null)
    setResults(null)
    setDisplayedText('')
    textToType.current = ''
    typingIndex.current = 0
    setTotalSteps(numSteps)

    try {
      // Build request payload (v2.0 with semantic steering)
      const params = {
        prompt,
        num_inference_steps: numSteps,
        guidance_scale: guidanceScale,
        intervention_active: interventionActive,
        intervention_strength: interventionStrength,
        intervention_step_start: interventionStart,
        intervention_step_end: interventionEnd,
        // v2.0 semantic parameters
        target_concept: targetConcept || undefined,
        injection_attribute: injectionAttribute || undefined,
        auto_detect_concepts: autoDetectConcepts
      }
      
      // Add initial log
      setLogs(prev => [...prev, 'Initializing Diffusion Detective...'])
      setLogs(prev => [...prev, `Prompt: "${params.prompt}"`])
      setLogs(prev => [...prev, `Steps: ${params.num_inference_steps} | Guidance: ${params.guidance_scale}`])
      
      if (params.intervention_active) {
        setLogs(prev => [...prev, `Intervention ACTIVE | Strength: ${params.intervention_strength}`])
        setLogs(prev => [...prev, `Intervention Zone: Steps ${params.intervention_step_end}-${params.intervention_step_start}`])
      } else {
        setLogs(prev => [...prev, 'Natural generation (no intervention)'])
      }
      
      setLogs(prev => [...prev, ''])
      setLogs(prev => [...prev, 'Generating images...'])
      setLogs(prev => [...prev, ''])

      // Simulate step progress for UI feedback
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + 1
          if (next >= numSteps) {
            clearInterval(stepInterval)
            return numSteps
          }
          return next
        })
      }, 120) // Slightly slower for better UX

      // Use simple endpoint (non-streaming for now)
      const response = await axios.post(`${API_BASE_URL}/generate_simple`, params, {
        timeout: 300000 // 5 minute timeout
      })

      clearInterval(stepInterval)
      setCurrentStep(numSteps)
      
      // Show analyzing phase for LLM reasoning
      setCurrentStatus('analyzing')
      setLogs(prev => [...prev, ''])
      setLogs(prev => [...prev, 'Extracting neural states...'])

      if (response.data.success) {
        processCompleteResponse(response.data)
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
            DIFFUSION DETECTIVE
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
              PROMPT
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
              INFERENCE STEPS: {numSteps}
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
              GUIDANCE SCALE: {guidanceScale}
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
              {/* v2.0: HYPOTHESIS TESTING PANEL */}
              <div className="border-b border-neon-cyan/30 pb-3 mb-3">
                <h3 className="text-xs font-bold text-neon-cyan mb-2">HYPOTHESIS TESTING (v2.0)</h3>
                
                <div className="space-y-2">
                  <div>
                    <label className="block text-xs font-bold mb-1 text-neon-cyan/80">
                      Focus Subject
                    </label>
                    <input
                      type="text"
                      value={targetConcept}
                      onChange={(e) => setTargetConcept(e.target.value)}
                      placeholder="e.g., tiger, mountain"
                      className="w-full px-2 py-1 text-xs bg-cyber-black border border-neon-cyan/50 rounded text-neon-cyan placeholder-neon-cyan/30"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-xs font-bold mb-1 text-neon-cyan/80">
                      ✨ Inject Attribute
                    </label>
                    <input
                      type="text"
                      value={injectionAttribute}
                      onChange={(e) => setInjectionAttribute(e.target.value)}
                      placeholder="e.g., neon, snowy, robot"
                      className="w-full px-2 py-1 text-xs bg-cyber-black border border-neon-cyan/50 rounded text-neon-cyan placeholder-neon-cyan/30"
                    />
                  </div>
                  
                  <label className="flex items-center space-x-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={autoDetectConcepts}
                      onChange={(e) => setAutoDetectConcepts(e.target.checked)}
                      className="w-3 h-3"
                    />
                    <span className="text-neon-cyan/70">Auto-detect concepts</span>
                  </label>
                  
                  {targetConcept && injectionAttribute && (
                    <div className="text-xs text-neon-cyan bg-neon-cyan/10 p-2 rounded border border-neon-cyan/30">
                      <span className="font-bold">WHAT IF:</span> "{targetConcept}" becomes "{injectionAttribute}"?
                    </div>
                  )}
                </div>
              </div>
              
              {/* ORIGINAL CONTROLS */}
              <div>
                <label className="block text-xs font-bold mb-2 text-neon-red">
                  INTERVENTION STRENGTH: {interventionStrength.toFixed(1)}x
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
                  START STEP: {interventionStart}
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
          {isGenerating ? 'ANALYZING...' : 'RUN ANALYSIS'}
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
                        <span className="text-xs text-neon-red font-bold mt-1">↓</span>
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
              <h3 className="text-sm text-neon-green/50">Ready for Analysis</h3>
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
                    {currentStatus === 'analyzing' ? '�️' : currentStatus === 'decoding' ? '🎨' : '�🔬'}
                  </motion.div>
                  <h2 className="text-2xl font-bold neon-glow mb-2"
                      style={{
                        color: currentStatus === 'analyzing' ? '#9b59b6' : 
                               currentStatus === 'decoding' ? '#f39c12' : '#00ff41'
                      }}>
                    {currentStatus === 'analyzing' ? '🧠 EXTRACTING NEURAL STATES' :
                     currentStatus === 'decoding' ? 'DECODING IMAGE' :
                     'ANALYSIS IN PROGRESS'}
                  </h2>
                  <p className="text-sm text-neon-green/70">
                    {currentStatus === 'complete' ? 'Complete!' :
                     currentStatus === 'analyzing' ? 'LLM analyzing attention patterns...' :
                     currentStatus === 'decoding' ? 'Rendering pixels...' :
                     `Step ${currentStep} / ${numSteps}`}
                  </p>
                  <div className="mt-4 w-64 mx-auto">
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full"
                        style={{
                          background: currentStatus === 'analyzing' ? 'linear-gradient(to right, #9b59b6, #e74c3c)' :
                                     currentStatus === 'decoding' ? 'linear-gradient(to right, #f39c12, #e67e22)' :
                                     'linear-gradient(to right, #00ff41, #00d4ff)'
                        }}
                        initial={{ width: '0%' }}
                        animate={{ 
                          width: currentStatus === 'analyzing' ? '99%' :
                                 currentStatus === 'decoding' ? '98%' :
                                 `${Math.min((currentStep / numSteps) * 97, 97)}%`
                        }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    <div className="text-xs text-neon-green/50 mt-2">
                      {currentStatus === 'analyzing' ? '99%' :
                       currentStatus === 'decoding' ? '98%' :
                       `${Math.floor((currentStep / numSteps) * 97)}%`}
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
          <div className={`mb-2 border-b pb-2 ${currentStatus === 'analyzing' ? 'border-purple-500 animate-pulse' : 'border-neon-green/30'}`}>
            <div className="flex items-center justify-between">
              <h3 className={`text-sm font-bold ${currentStatus === 'analyzing' ? 'text-purple-500 animate-pulse' : 'text-neon-green'}`}>
                🕵️ DETECTIVE'S LOG {currentStatus === 'analyzing' && '✍️'}
              </h3>
              <span className={`text-xs ${
                currentStatus === 'analyzing' ? 'text-purple-500 animate-pulse' :
                isGenerating ? 'text-neon-green animate-pulse' : 
                'text-gray-600'
              }`}>
                {currentStatus === 'analyzing' ? '● WRITING' : 
                 isGenerating ? '● ACTIVE' : 
                 '○ IDLE'}
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
            
            {/* v2.0: Multi-Concept Focus Bar Chart */}
            {Object.keys(focusScores).length > 0 && (
              <div className="mt-2 pt-2 border-t border-neon-purple/20">
                <div className="text-xs text-neon-purple/70 mb-1">📊 Multi-Concept Balance:</div>
                <div className="space-y-1">
                  {Object.entries(focusScores).map(([concept, data]) => {
                    const confidence = data.confidence || 0
                    const barWidth = `${Math.min(confidence, 100)}%`
                    const isNoun = data.category === 'noun'
                    const barColor = isNoun ? 'bg-neon-cyan' : 'bg-neon-purple'
                    
                    return (
                      <div key={concept} className="flex items-center gap-2">
                        <span className="text-neon-purple/80 text-xs font-mono w-20 truncate">
                          {concept.toUpperCase()}
                        </span>
                        <div className="flex-1 bg-cyber-dark rounded h-3 relative overflow-hidden border border-neon-purple/20">
                          <motion.div
                            className={`h-full ${barColor} opacity-70`}
                            initial={{ width: 0 }}
                            animate={{ width: barWidth }}
                            transition={{ duration: 0.5 }}
                          >
                            <div className={`absolute inset-0 ${barColor.replace('bg-', 'shadow-')} opacity-50 blur-sm`} />
                          </motion.div>
                          <div className="absolute right-1 top-0 h-full flex items-center text-white text-xs font-bold">
                            {Math.round(confidence)}%
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
                <div className="text-xs text-neon-purple/50 mt-1">
                  <span className="text-neon-cyan">█</span> Subjects | <span className="text-neon-purple">█</span> Attributes
                </div>
              </div>
            )}
          </div>
          
          <div className="space-y-2">
            {logs.map((log, idx) => {
              // Handle structured log objects (new LLM format)
              if (typeof log === 'object' && log.isStructured) {
                // Color coding based on type
                let textColor = 'text-neon-green/90'
                let pulseEffect = ''
                let borderColor = 'border-neon-green/20'
                
                if (log.type === 'injection_start') {
                  textColor = 'text-red-500 font-bold'
                  pulseEffect = 'animate-pulse'
                  borderColor = 'border-red-500/50'
                } else if (log.type === 'conflict') {
                  textColor = 'text-yellow-400 font-semibold'
                  borderColor = 'border-yellow-400/30'
                } else if (log.type === 'collateral_damage') {
                  textColor = 'text-red-400/70'
                  borderColor = 'border-red-400/30'
                }
                
                return (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className={`border-l-2 ${borderColor} pl-3 py-2 mb-2`}
                  >
                    {/* Main message with color coding and proper text wrapping */}
                    <div className={`${textColor} ${pulseEffect} mb-2 leading-relaxed break-words`}>
                      <span className="font-mono text-sm opacity-80">[{log.range}]</span>{' '}
                      <span className="inline-block">{log.message}</span>
                    </div>
                    
                    {/* Mini bar charts for stats with delta indicators */}
                    {log.stats && Object.keys(log.stats).length > 0 && (
                      <div className="ml-2 space-y-0.5 mt-1">
                        {Object.entries(log.stats).slice(0, 4).map(([concept, value]) => {
                          const percentage = Math.min(Math.max(value, 0), 100)
                          const barLength = Math.floor(percentage / 10)
                          const bar = '█'.repeat(barLength) + '░'.repeat(10 - barLength)
                          
                          // Check if baseline comparison exists for this concept
                          const comparison = log.baseline_comparison?.[concept]
                          let deltaIndicator = ''
                          let deltaColor = 'text-gray-500'
                          
                          if (comparison) {
                            const percentChange = comparison.percent_change
                            if (Math.abs(percentChange) < 5) {
                              deltaIndicator = '→'
                              deltaColor = 'text-blue-400'
                            } else if (percentChange > 0) {
                              deltaIndicator = '↑'
                              deltaColor = 'text-green-400'
                            } else {
                              deltaIndicator = '↓'
                              deltaColor = 'text-red-400'
                            }
                            
                            const changeText = `${percentChange > 0 ? '+' : ''}${percentChange.toFixed(0)}%`
                            deltaIndicator += ` ${changeText}`
                          }
                          
                          return (
                            <div key={concept} className="text-xs text-neon-cyan/80 font-mono flex items-center gap-2">
                              <span className="w-20 truncate text-neon-purple/70">
                                {concept.toUpperCase()}
                              </span>
                              <span className="text-neon-green/60">[{bar}]</span>
                              <span className="text-neon-cyan/90">{percentage.toFixed(1)}%</span>
                              {deltaIndicator && (
                                <span className={`${deltaColor} text-xs ml-1`}>
                                  {deltaIndicator}
                                </span>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </motion.div>
                )
              }
              
              // Handle string logs (old format)
              const isInjection = typeof log === 'string' && log.includes('[INJECTION]')
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={isInjection ? 'text-neon-red font-bold' : 'text-neon-green/90'}
                  dangerouslySetInnerHTML={{ __html: highlightKeywords(typeof log === 'string' ? log : JSON.stringify(log)) }}
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
