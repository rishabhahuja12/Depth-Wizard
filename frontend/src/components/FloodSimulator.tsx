import React, { useMemo } from 'react';
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
  // Precompute 512-bin cumulative distribution function (CDF) for O(1) slider evaluation
  const { cdf, totalPixels } = useMemo(() => {
    if (!dsmRaw || dsmRaw.length === 0) return { cdf: null, totalPixels: 0 };
    const numBins = 512;
    const counts = new Uint32Array(numBins);
    let total = 0;
    const range = elevationMax - elevationMin;
    const scale = range > 1e-6 ? (numBins - 1) / range : 0;

    for (let i = 0; i < dsmRaw.length; i++) {
      const row = dsmRaw[i];
      for (let j = 0; j < row.length; j++) {
        const val = row[j];
        const bin = Math.max(0, Math.min(numBins - 1, Math.floor((val - elevationMin) * scale)));
        counts[bin]++;
        total++;
      }
    }

    if (total === 0) return { cdf: null, totalPixels: 0 };

    const cdfArray = new Float32Array(numBins);
    let accum = 0;
    for (let b = 0; b < numBins; b++) {
      accum += counts[b];
      cdfArray[b] = (accum / total) * 100;
    }
    return { cdf: cdfArray, totalPixels: total };
  }, [dsmRaw, elevationMin, elevationMax]);

  // O(1) lookup of inundated percentage on slider tick with continuous piecewise linear interpolation
  const floodedPercent = useMemo(() => {
    if (!cdf || totalPixels === 0) return 0;
    if (waterLevel <= elevationMin) return 0;
    if (waterLevel >= elevationMax) return 100;
    const numBins = cdf.length;
    const range = elevationMax - elevationMin;
    if (range < 1e-6) return 0;
    const norm = Math.max(0, Math.min(1, (waterLevel - elevationMin) / range));
    const floatBin = norm * (numBins - 1);
    const i0 = Math.floor(floatBin);
    const frac = floatBin - i0;
    const c0 = i0 > 0 ? cdf[i0 - 1] : 0;
    const c1 = cdf[i0];
    return Math.min(100, Math.max(0, c0 + (c1 - c0) * frac));
  }, [cdf, waterLevel, elevationMin, elevationMax, totalPixels]);

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

      {waterLevel > elevationMin && (
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-blue-400">{floodedPercent.toFixed(1)}%</p>
          <p className="text-xs text-slate-400">Area Inundated</p>
        </div>
      )}
    </div>
  );
}
