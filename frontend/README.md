# Diffusion Detective Frontend

A modern, cyberpunk-themed React interface for the Diffusion Detective API.

## Setup

1. **Install Dependencies**
```bash
npm install
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env if your backend runs on a different URL
```

3. **Run Development Server**
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Build for Production

```bash
npm run build
npm run preview
```

## Features

✨ Dark cyberpunk theme with neon accents
✨ Real-time terminal-style log display
✨ Interactive image comparison slider
✨ Smooth animations with Framer Motion
✨ Responsive design
✨ Typewriter effects for investigation reports

## Components

- **ControlPanel**: Mission control for generation parameters
- **Timeline**: Visual progress bar with intervention zone
- **Terminal**: Scrolling log display with syntax highlighting
- **ComparisonSlider**: Side-by-side image comparison

## Technology Stack

- React 18
- Vite
- Tailwind CSS
- Framer Motion
- react-compare-image
- Axios
