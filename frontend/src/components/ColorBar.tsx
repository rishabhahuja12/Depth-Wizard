import React from 'react';

interface ColorBarProps {
  min: number;
  max: number;
  unit: string;
}

export default function ColorBar({ min, max, unit }: ColorBarProps) {
  return (
    <div className="neo-box p-2.5 flex flex-col items-center gap-1.5 shadow-[4px_4px_0px_0px_#000]">
      <div className="text-[9px] font-black uppercase tracking-wider text-[#FFE600]">TURBO DSM</div>
      <span className="text-[11px] text-[#FF3366] font-mono font-black">{max.toFixed(1)}</span>
      <div
        className="w-4 h-32 border-2 border-black"
        style={{
          background: 'linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff)',
        }}
      />
      <span className="text-[11px] text-[#00FF88] font-mono font-black">{min.toFixed(1)}</span>
      <span className="text-[9px] font-mono font-bold text-white/60 bg-black px-1 border border-white/20">
        {unit}
      </span>
    </div>
  );
}
