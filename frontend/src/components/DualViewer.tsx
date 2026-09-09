import React from 'react';

interface DualViewerProps {
  rgbB64: string;
  dsmColorizedB64: string;
  elevationMin: number;
  elevationMax: number;
  unit: string;
}

export default function DualViewer({ rgbB64, dsmColorizedB64, elevationMin, elevationMax, unit }: DualViewerProps) {
  return (
    <div className="glass-panel p-4 space-y-3">
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">2D Comparison</h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <p className="text-xs text-slate-400 text-center">Original RGB</p>
          <img
            src={`data:image/jpeg;base64,${rgbB64}`}
            alt="RGB"
            className="w-full rounded-lg border border-white/5"
          />
        </div>
        <div className="space-y-2">
          <p className="text-xs text-slate-400 text-center">Estimated DSM</p>
          <img
            src={`data:image/png;base64,${dsmColorizedB64}`}
            alt="DSM"
            className="w-full rounded-lg border border-white/5"
          />
        </div>
      </div>
      <div className="flex justify-between text-xs text-slate-500">
        <span>{elevationMin.toFixed(1)} {unit}</span>
        <span className="text-slate-400">Elevation Range</span>
        <span>{elevationMax.toFixed(1)} {unit}</span>
      </div>
    </div>
  );
}
