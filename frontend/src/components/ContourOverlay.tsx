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
        className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all w-full
          ${active
            ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
            : 'bg-white/5 text-slate-400 hover:bg-white/10 border border-white/5'
          }`}
      >
        <Map className="w-4 h-4" />
        Contour Lines
        <span className="ml-auto text-xs opacity-60">{active ? 'ON' : 'OFF'}</span>
      </button>

      {active && (
        <div className="pl-2 space-y-1">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Interval</span>
            <span className="font-mono">{interval.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min={0.5}
            max={20}
            step={0.5}
            value={interval}
            onChange={(e) => onIntervalChange(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
}
