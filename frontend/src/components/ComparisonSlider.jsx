import { useState } from 'react'
import { motion } from 'framer-motion'
import ReactCompareImage from 'react-compare-image'

const ComparisonSlider = ({ naturalImage, controlledImage, metadata }) => {
  const [showMetadata, setShowMetadata] = useState(false)

  // Debug logging
  console.log('ComparisonSlider props:', {
    hasNatural: !!naturalImage,
    hasControlled: !!controlledImage,
    hasMetadata: !!metadata,
    naturalPrefix: naturalImage?.substring(0, 50),
    controlledPrefix: controlledImage?.substring(0, 50)
  })

  // Ensure images have proper data URI format
  const naturalImageSrc = naturalImage?.startsWith('data:') ? naturalImage : `data:image/png;base64,${naturalImage}`
  const controlledImageSrc = controlledImage?.startsWith('data:') ? controlledImage : `data:image/png;base64,${controlledImage}`

  return (
    <div className="terminal-window">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 border-b border-neon-green/30 pb-2">
        <h3 className="text-lg font-bold text-neon-green neon-glow">
          🔬 COMPARATIVE ANALYSIS
        </h3>
        <button
          onClick={() => setShowMetadata(!showMetadata)}
          className="text-xs cyber-button py-1"
        >
          {showMetadata ? 'Hide' : 'Show'} Metadata
        </button>
      </div>

      {/* Metadata Panel */}
      {showMetadata && metadata && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-4 bg-cyber-black/50 border border-neon-green/30 rounded p-3 text-xs"
        >
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <div>
              <span className="text-neon-green/60">Prompt:</span>
              <p className="text-neon-green/90 mt-1 break-words">{metadata.prompt}</p>
            </div>
            <div>
              <span className="text-neon-green/60">Steps:</span>
              <p className="text-neon-green/90 mt-1">{metadata.num_inference_steps}</p>
            </div>
            <div>
              <span className="text-neon-green/60">Guidance Scale:</span>
              <p className="text-neon-green/90 mt-1">{metadata.guidance_scale}</p>
            </div>
            <div>
              <span className="text-neon-green/60">Intervention:</span>
              <p className={`mt-1 font-bold ${metadata.intervention_active ? 'text-neon-red' : 'text-neon-green/70'}`}>
                {metadata.intervention_active ? 'ACTIVE' : 'INACTIVE'}
              </p>
            </div>
            {metadata.intervention_active && (
              <>
                <div>
                  <span className="text-neon-green/60">Strength:</span>
                  <p className="text-neon-green/90 mt-1">{metadata.intervention_strength}</p>
                </div>
                <div>
                  <span className="text-neon-green/60">Range:</span>
                  <p className="text-neon-green/90 mt-1">{metadata.intervention_range}</p>
                </div>
              </>
            )}
          </div>
        </motion.div>
      )}

      {/* Comparison Labels */}
      <div className="flex justify-between mb-2 text-xs text-neon-green/70">
        <span className="flex items-center">
          <span className="w-2 h-2 bg-neon-blue rounded-full mr-2" />
          Natural (Baseline)
        </span>
        <span className="flex items-center">
          <span className="w-2 h-2 bg-neon-purple rounded-full mr-2" />
          Controlled (Intervened)
        </span>
      </div>

      {/* Image Comparison Slider */}
      <div className="relative rounded-lg overflow-hidden border border-neon-green/50">
        {naturalImageSrc && controlledImageSrc ? (
          <ReactCompareImage
            leftImage={naturalImageSrc}
            rightImage={controlledImageSrc}
            sliderLineColor="#00FF41"
            sliderLineWidth={3}
            handleSize={40}
            hover={true}
          />
        ) : (
          <div className="w-full h-96 flex items-center justify-center text-neon-green/50">
            <p>Loading images...</p>
          </div>
        )}
        
        {/* Overlay Labels */}
        <div className="absolute top-4 left-4 bg-cyber-black/90 border border-neon-blue px-3 py-1 rounded text-xs text-neon-blue font-bold">
          NATURAL
        </div>
        <div className="absolute top-4 right-4 bg-cyber-black/90 border border-neon-purple px-3 py-1 rounded text-xs text-neon-purple font-bold">
          CONTROLLED
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-3 text-xs text-center text-neon-green/50">
        💡 Drag the slider to compare • Hover to preview both sides
      </div>

      {/* Download Buttons */}
      <div className="mt-4 flex gap-3 justify-center">
        <a
          href={naturalImageSrc}
          download="natural_image.png"
          className="cyber-button text-xs"
        >
          ⬇️ Download Natural
        </a>
        <a
          href={controlledImageSrc}
          download="controlled_image.png"
          className="cyber-button text-xs"
        >
          ⬇️ Download Controlled
        </a>
      </div>

    </div>
  )
}

export default ComparisonSlider
