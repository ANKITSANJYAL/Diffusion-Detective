/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cyber-black': '#050505',
        'cyber-dark': '#0a0a0a',
        'cyber-gray': '#1a1a1a',
        'neon-green': '#00FF41',
        'neon-red': '#FF0055',
        'neon-blue': '#00D9FF',
        'neon-purple': '#BD00FF',
      },
      fontFamily: {
        'mono': ['"Fira Code"', 'monospace'],
      },
      animation: {
        'typewriter': 'typewriter 2s steps(40) forwards',
        'blink': 'blink 1s infinite',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'scan': 'scan 2s linear infinite',
      },
      keyframes: {
        typewriter: {
          '0%': { width: '0' },
          '100%': { width: '100%' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        'pulse-glow': {
          '0%, 100%': { 
            boxShadow: '0 0 10px rgba(0, 255, 65, 0.5), 0 0 20px rgba(0, 255, 65, 0.3)',
          },
          '50%': { 
            boxShadow: '0 0 20px rgba(0, 255, 65, 0.8), 0 0 40px rgba(0, 255, 65, 0.5)',
          },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
    },
  },
  plugins: [],
}
