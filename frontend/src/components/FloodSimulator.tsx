import React, { useMemo } from 'react';
import { Waves, AlertTriangle } from 'lucide-react';

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
    <div className="neo-card">
      {/* Header */}
      <div className="neo-card-header bg-[#00F0FF] text-black">
        <span className="flex items-center gap-1.5 font-black">
          <Waves className="w-4 h-4" /> Flood Inundation Simulator
        </span>
        <span className="font-mono text-[10px] bg-black text-[#00F0FF] px-1.5 py-0.5 border border-black">
          UVP // DISASTER
        </span>
      </div>

      <div className="p-3.5 space-y-3.5">
        {/* Slider Controls */}
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs font-mono font-bold">
            <span className="text-white/80 uppercase">Water Altitude</span>
            <span className="bg-[#FFE600] text-black px-2 py-0.5 border border-black font-black">
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
            className="neo-slider"
          />

          <div className="flex justify-between text-[10px] font-mono text-white/50">
            <span>MIN: {elevationMin.toFixed(1)}</span>
            <span>MAX: {elevationMax.toFixed(1)}</span>
          </div>
        </div>

        {/* Inundated Percentage Box */}
        <div className="bg-[#00F0FF] text-black border-2 border-black p-3 shadow-[3px_3px_0px_0px_#000]">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-3xl font-black tracking-tight leading-none">
                {floodedPercent.toFixed(1)}%
              </p>
              <p className="text-[10px] font-mono font-black uppercase tracking-wider mt-1 text-black/80">
                Terrain Submerged
              </p>
            </div>
            {floodedPercent > 30 && (
              <div className="bg-[#FF3366] text-white p-1.5 border-2 border-black">
                <AlertTriangle className="w-5 h-5 animate-pulse" />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
