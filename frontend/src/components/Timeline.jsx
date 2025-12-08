import { motion } from 'framer-motion'

const Timeline = ({ progress, totalSteps, interventionRange }) => {
  const currentStep = Math.floor((progress / 100) * totalSteps)
  
  // Calculate intervention zone positions
  const interventionStartPercent = (interventionRange.start / totalSteps) * 100
  const interventionEndPercent = (interventionRange.end / totalSteps) * 100

  return (
    <div className="terminal-window">
      <div className="mb-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-bold text-neon-green/90">
            📊 GENERATION TIMELINE
          </h3>
          <span className="text-xs text-neon-green/70">
            Step {currentStep} / {totalSteps}
          </span>
        </div>
        
        <div className="text-xs text-neon-green/60 mb-2">
          Progress: {progress.toFixed(1)}%
        </div>
      </div>

      {/* Progress Bar Container */}
      <div className="relative h-10 bg-cyber-black border border-neon-green/30 rounded overflow-hidden">
        
        {/* Intervention Zone Indicator */}
        <div
          className="absolute h-full bg-neon-red/20 border-x border-neon-red/50"
          style={{
            left: `${interventionEndPercent}%`,
            right: `${100 - interventionStartPercent}%`
          }}
        >
          <div className="absolute inset-0 flex items-center justify-center text-xs text-neon-red/80 font-bold">
            INTERVENTION ZONE
          </div>
        </div>

        {/* Progress Fill */}
        <motion.div
          className="absolute h-full bg-gradient-to-r from-neon-green/30 to-neon-green/60"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.3 }}
        />

        {/* Current Position Indicator */}
        <motion.div
          className="absolute h-full w-1 bg-neon-green shadow-lg shadow-neon-green"
          initial={{ left: 0 }}
          animate={{ left: `${progress}%` }}
          transition={{ duration: 0.3 }}
        >
          <div className="absolute -top-1 -left-2 w-4 h-4 bg-neon-green rounded-full animate-pulse" />
        </motion.div>

        {/* Step Markers */}
        <div className="absolute inset-0 flex items-center justify-between px-1">
          {[0, 25, 50, 75, 100].map((marker) => (
            <div
              key={marker}
              className="text-xs text-neon-green/40 font-bold"
              style={{ fontSize: '0.65rem' }}
            >
              {Math.floor((marker / 100) * totalSteps)}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-neon-green/60 rounded mr-1" />
            <span className="text-neon-green/70">Progress</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-neon-red/60 rounded mr-1" />
            <span className="text-neon-red/70">Intervention</span>
          </div>
        </div>
        <div className="text-neon-green/50">
          Steps {interventionRange.end}-{interventionRange.start}
        </div>
      </div>
    </div>
  )
}

export default Timeline
