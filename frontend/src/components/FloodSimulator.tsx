import React from 'react';
import { Waves } from 'lucide-react';

interface FloodSimulatorProps {
  waterLevel: number;
  onWaterLevelChange: (level: number) => void;
  elevationMin: number;
  elevationMax: number;
  unit: string;
  dsmRaw: number[][];
}

export default function FloodSimulator({
  waterLevel, onWaterLevelChange, elevationMin, elevationMax, unit, dsmRaw
}: FloodSimulatorProps) {
  // Calculate flooded percentage
  let floodedPercent = 0;
  if (dsmRaw && dsmRaw.length > 0 && waterLevel > 0) {
    let total = 0;
    let flooded = 0;
    for (const row of dsmRaw) {
      for (const val of row) {
        total++;
        if (val < waterLevel) flooded++;
      }
    }
    floodedPercent = total > 0 ? (flooded / total) * 100 : 0;
  }

  return (
    <div className="glass-panel p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Waves className="w-4 h-4 text-blue-400" />
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Flood Simulator
        </h3>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Water Level</span>
          <span className="text-blue-400 font-mono">
            {waterLevel.toFixed(1)} {unit}
          </span>
        </div>
        <input
          type="range"
          min={elevationMin}
          max={elevationMax}
          step={Math.max((elevationMax - elevationMin) / 200, 0.01)}
          value={waterLevel}
          onChange={(e) => onWaterLevelChange(parseFloat(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-slate-500">
          <span>{elevationMin.toFixed(1)}</span>
          <span>{elevationMax.toFixed(1)}</span>
        </div>
      </div>

      {waterLevel > 0 && (
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-blue-400">{floodedPercent.toFixed(1)}%</p>
          <p className="text-xs text-slate-400">Area Inundated</p>
        </div>
      )}
    </div>
  );
}
