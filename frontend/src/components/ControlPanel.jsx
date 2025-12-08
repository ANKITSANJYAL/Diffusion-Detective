import { useState } from 'react'
import { motion } from 'framer-motion'

const ControlPanel = ({ onGenerate, isGenerating }) => {
  const [prompt, setPrompt] = useState('A majestic lion standing on a mountain peak at sunset')
  const [numSteps, setNumSteps] = useState(50)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [interventionActive, setInterventionActive] = useState(true)
  const [interventionStrength, setInterventionStrength] = useState(1.0)
  const [interventionStart, setInterventionStart] = useState(40)
  const [interventionEnd, setInterventionEnd] = useState(20)
  const [seed, setSeed] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    
    const params = {
      prompt,
      num_inference_steps: parseInt(numSteps),
      guidance_scale: parseFloat(guidanceScale),
      intervention_active: interventionActive,
      intervention_strength: parseFloat(interventionStrength),
      intervention_step_start: parseInt(interventionStart),
      intervention_step_end: parseInt(interventionEnd),
      seed: seed ? parseInt(seed) : null
    }
    
    onGenerate(params)
  }

  return (
    <div className="terminal-window scanline">
      <div className="flex items-center justify-between mb-4 border-b border-neon-green/30 pb-2">
        <h2 className="text-xl font-bold neon-glow">⚙️ MISSION CONTROL</h2>
        <div className="text-xs text-neon-green/60">
          {isGenerating ? '● ACTIVE' : '○ STANDBY'}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        
        {/* Prompt Input */}
        <div>
          <label className="block text-sm mb-1 text-neon-green/80">
            📝 Prompt
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full bg-cyber-black border border-neon-green/30 rounded px-3 py-2 
                     text-neon-green focus:border-neon-green focus:outline-none
                     focus:ring-1 focus:ring-neon-green"
            rows="3"
            placeholder="Describe the image you want to generate..."
            disabled={isGenerating}
            required
          />
        </div>

        {/* Generation Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          <div>
            <label className="block text-sm mb-1 text-neon-green/80">
              🔢 Inference Steps: {numSteps}
            </label>
            <input
              type="range"
              min="20"
              max="100"
              value={numSteps}
              onChange={(e) => setNumSteps(e.target.value)}
              className="w-full"
              disabled={isGenerating}
            />
          </div>

          <div>
            <label className="block text-sm mb-1 text-neon-green/80">
              🎚️ Guidance Scale: {guidanceScale}
            </label>
            <input
              type="range"
              min="1"
              max="20"
              step="0.5"
              value={guidanceScale}
              onChange={(e) => setGuidanceScale(e.target.value)}
              className="w-full"
              disabled={isGenerating}
            />
          </div>

        </div>

        {/* Intervention Toggle */}
        <div className="border border-neon-green/30 rounded p-4 bg-cyber-dark/50">
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm font-bold text-neon-green/90">
              🧪 LATENT STEERING INTERVENTION
            </label>
            <button
              type="button"
              onClick={() => setInterventionActive(!interventionActive)}
              className={`px-4 py-1 rounded text-sm font-bold transition-all ${
                interventionActive 
                  ? 'bg-neon-green text-cyber-black' 
                  : 'bg-cyber-gray text-neon-green border border-neon-green'
              }`}
              disabled={isGenerating}
            >
              {interventionActive ? 'ON' : 'OFF'}
            </button>
          </div>

          {interventionActive && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="space-y-3"
            >
              
              <div>
                <label className="block text-sm mb-1 text-neon-green/80">
                  💪 Intervention Strength: {interventionStrength.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={interventionStrength}
                  onChange={(e) => setInterventionStrength(e.target.value)}
                  className="w-full"
                  disabled={isGenerating}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm mb-1 text-neon-green/80">
                    🔽 Start Step: {interventionStart}
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={interventionStart}
                    onChange={(e) => setInterventionStart(e.target.value)}
                    className="w-full bg-cyber-black border border-neon-green/30 rounded px-2 py-1 
                             text-neon-green text-sm"
                    disabled={isGenerating}
                  />
                </div>

                <div>
                  <label className="block text-sm mb-1 text-neon-green/80">
                    🔼 End Step: {interventionEnd}
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="100"
                    value={interventionEnd}
                    onChange={(e) => setInterventionEnd(e.target.value)}
                    className="w-full bg-cyber-black border border-neon-green/30 rounded px-2 py-1 
                             text-neon-green text-sm"
                    disabled={isGenerating}
                  />
                </div>
              </div>

            </motion.div>
          )}
        </div>

        {/* Seed Input */}
        <div>
          <label className="block text-sm mb-1 text-neon-green/80">
            🌱 Random Seed (optional, for reproducibility)
          </label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            className="w-full bg-cyber-black border border-neon-green/30 rounded px-3 py-2 
                     text-neon-green focus:border-neon-green focus:outline-none"
            placeholder="Leave empty for random"
            disabled={isGenerating}
          />
        </div>

        {/* Submit Button */}
        <motion.button
          type="submit"
          disabled={isGenerating}
          className={`w-full py-3 rounded font-bold text-lg transition-all ${
            isGenerating
              ? 'bg-cyber-gray text-neon-green/50 cursor-not-allowed'
              : 'cyber-button animate-pulse-glow'
          }`}
          whileHover={!isGenerating ? { scale: 1.02 } : {}}
          whileTap={!isGenerating ? { scale: 0.98 } : {}}
        >
          {isGenerating ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              ANALYZING...
            </span>
          ) : (
            '🚀 RUN ANALYSIS'
          )}
        </motion.button>

      </form>
    </div>
  )
}

export default ControlPanel
