import React from 'react';
import TerrainCanvas from './TerrainCanvas.tsx';

export interface MinimapProps {
  heightmapB64: string;
  rgbB64: string;
  normalMapB64?: string;
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
  };
  verticalScale: number;
  waterLevel: number;
  showContours?: boolean;
  contourInterval?: number;
  dsmRaw: number[][];
  className?: string;
}

/**
 * Minimap component (§3.6): A secondary small (220x160px) viewport
 * reusing TerrainCanvas with a locked, non-interactive camera.
 */
export default function Minimap({
  showContours = false,
  className = '',
  ...props
}: MinimapProps) {
  return (
    <div
      className={`w-[220px] h-[160px] bg-[#050608]/90 border border-white/20 shadow-2xl backdrop-blur-md overflow-hidden relative select-none ${className}`}
    >
      {/* Header Overlay Label */}
      <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5 bg-black/80 px-2 py-0.5 border border-white/15 text-[9px] font-mono font-bold tracking-widest text-neutral-300 uppercase pointer-events-none">
        <span className="w-1.5 h-1.5 bg-[#38BDF8] animate-pulse" />
        SMOOTH OVERVIEW
      </div>

      {/* Non-interactive 3D canvas */}
      <div className="w-full h-full pointer-events-none">
        <TerrainCanvas
          {...props}
          interactive={false}
          showContours={showContours}
        />
      </div>
    </div>
  );
}
