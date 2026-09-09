import React from 'react';
import { Map } from 'lucide-react';

interface ContourOverlayProps {
  active: boolean;
  onToggle: () => void;
  interval: number;
  onIntervalChange: (val: number) => void;
}

export default function ContourOverlay({ active, onToggle, interval, onIntervalChange }: ContourOverlayProps) {
  return (
    <div className="space-y-2">
      <button
        onClick={onToggle}
        className={`w-full neo-btn text-xs py-2 px-3 flex items-center justify-between transition-colors cursor-pointer ${
          active ? 'bg-[#FFE600] text-black shadow-[3px_3px_0px_0px_#000]' : 'bg-[#1A1D24] text-white hover:bg-white/10'
        }`}
      >
        <span className="flex items-center gap-1.5 font-black">
          <Map className="w-3.5 h-3.5" /> Topographic Contours
        </span>
        <span
          className={`font-mono text-[10px] font-black px-1.5 py-0.5 border ${
            active ? 'bg-black text-[#FFE600] border-black' : 'bg-black/60 text-white/70 border-white/20'
          }`}
        >
          {active ? 'ACTIVE' : 'OFF'}
        </span>
      </button>

      {active && (
        <div className="bg-black/50 border-2 border-black p-2.5 space-y-1.5">
          <div className="flex justify-between text-[11px] font-mono font-bold text-white/80">
            <span>Interval Step</span>
            <span className="text-[#FFE600]">{interval.toFixed(1)}m</span>
          </div>
          <input
            type="range"
            min={0.5}
            max={20}
            step={0.5}
            value={interval}
            onChange={(e) => onIntervalChange(parseFloat(e.target.value))}
            className="neo-slider"
          />
          <div className="flex justify-between text-[9px] font-mono text-white/40">
            <span>0.5m</span>
            <span>20.0m</span>
          </div>
        </div>
      )}
    </div>
  );
}
