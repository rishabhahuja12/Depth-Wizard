import React from 'react';

interface ColorBarProps {
  min: number;
  max: number;
  unit: string;
}

export default function ColorBar({ min, max, unit }: ColorBarProps) {
  return (
    <div className="flex flex-col items-center gap-1 glass-panel px-2 py-3">
      <span className="text-xs text-slate-300 font-mono font-medium">{max.toFixed(1)}</span>
      <div
        className="w-3 h-28 rounded-full border border-white/20 shadow-inner"
        style={{
          background: 'linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff)',
        }}
      />
      <span className="text-xs text-slate-300 font-mono font-medium">{min.toFixed(1)}</span>
      <span className="text-[10px] text-slate-400 mt-1">{unit}</span>
    </div>
  );
}
