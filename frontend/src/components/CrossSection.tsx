import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Scissors, X } from 'lucide-react';

interface CrossSectionProps {
  dsmRaw: number[][];
  unit: string;
  active: boolean;
  onToggle: () => void;
  pixelSize?: number;
}

interface ProfilePoint {
  distance: number;
  elevation: number;
}

export default function CrossSection({ dsmRaw, unit, active, onToggle, pixelSize }: CrossSectionProps) {
  // Horizontal cross-section through center of the elevation map
  const isMetric = unit === 'meters' && pixelSize !== undefined && pixelSize > 0;
  const distScale = isMetric ? pixelSize! : 1.0;
  const distUnit = isMetric ? 'm' : 'px';

  const profileData = useMemo<ProfilePoint[]>(() => {
    if (!dsmRaw || dsmRaw.length === 0) return [];

    const h = dsmRaw.length;
    const w = dsmRaw[0]?.length || 0;
    if (w === 0) return [];

    const midRow = Math.floor(h / 2);
    const K = Math.min(w, 256);
    const points: ProfilePoint[] = [];

    for (let k = 0; k < K; k++) {
      const col = Math.floor((k / (K - 1)) * (w - 1));
      const elevation = dsmRaw[midRow]?.[col] ?? 0;
      const distance = (k / (K - 1)) * (w * distScale);
      points.push({ distance: Math.round(distance * 10) / 10, elevation: Math.round(elevation * 100) / 100 });
    }

    return points;
  }, [dsmRaw, distScale]);

  if (!active || profileData.length === 0) {
    return (
      <div className="neo-card">
        <button
          onClick={onToggle}
          className="w-full bg-[#1A1D24] text-white hover:bg-[#FFE600] hover:text-black font-black text-xs uppercase tracking-wider p-3 flex items-center justify-between transition-colors cursor-pointer"
        >
          <span className="flex items-center gap-2">
            <Scissors className="w-4 h-4 text-[#FF3366]" /> Cross-Section Profile
          </span>
          <span className="bg-black text-white text-[10px] px-2 py-0.5 border border-white/30 font-mono">
            EXPAND
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="neo-card">
      <div className="neo-card-header bg-[#FF3366] text-white">
        <span className="flex items-center gap-1.5 font-black">
          <Scissors className="w-4 h-4" /> Elevation Transect
        </span>
        <button
          onClick={onToggle}
          className="bg-black text-white p-1 hover:bg-white hover:text-black transition-colors cursor-pointer"
          title="Close Transect"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="p-3 space-y-2">
        <div className="flex justify-between items-center text-[10px] font-mono text-white/70">
          <span>Center Axis Transect</span>
          <span className="text-[#FFE600] font-bold">
            {isMetric ? `${distScale.toFixed(2)}m GSD` : 'Relative Coordinates'}
          </span>
        </div>

        <div className="h-44 bg-black border-2 border-black p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={profileData}>
              <CartesianGrid strokeDasharray="2 2" stroke="#262b38" />
              <XAxis
                dataKey="distance"
                stroke="#8c97ab"
                fontSize={10}
                tickFormatter={(v) => `${Number(v).toFixed(0)}${distUnit}`}
              />
              <YAxis
                stroke="#8c97ab"
                fontSize={10}
                tickFormatter={(v) => `${Number(v).toFixed(1)}`}
              />
              <Tooltip
                contentStyle={{
                  background: '#12151B',
                  border: '2px solid #000000',
                  boxShadow: '3px 3px 0px 0px #000',
                  color: '#ffffff',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                }}
                formatter={(value: any) => [`${Number(value).toFixed(2)} ${unit}`, 'Height']}
                labelFormatter={(label) => `Distance: ${label} ${distUnit}`}
              />
              <Line
                type="monotone"
                dataKey="elevation"
                stroke="#00F0FF"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5, fill: '#FFE600', stroke: '#000000', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
