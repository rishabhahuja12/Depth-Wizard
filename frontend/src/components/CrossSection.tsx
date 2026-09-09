import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Scissors } from 'lucide-react';

interface CrossSectionProps {
  dsmRaw: number[][];
  unit: string;
  active: boolean;
  onToggle: () => void;
}

interface ProfilePoint {
  distance: number;
  elevation: number;
}

export default function CrossSection({ dsmRaw, unit, active, onToggle }: CrossSectionProps) {
  // Horizontal cross-section through center of the elevation map
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
      const distance = (k / (K - 1)) * w;
      points.push({ distance: Math.round(distance * 10) / 10, elevation: Math.round(elevation * 100) / 100 });
    }

    return points;
  }, [dsmRaw]);

  if (!active || profileData.length === 0) {
    return (
      <div className="glass-panel p-4">
        <button
          onClick={onToggle}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all w-full
            ${active
              ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
              : 'bg-white/5 text-slate-400 hover:bg-white/10 border border-white/5'
            }`}
        >
          <Scissors className="w-4 h-4" />
          Cross-Section Profile
        </button>
      </div>
    );
  }

  return (
    <div className="glass-panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Scissors className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Elevation Profile
          </h3>
        </div>
        <button
          onClick={onToggle}
          className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          Close
        </button>
      </div>

      <p className="text-xs text-slate-500">
        Horizontal cross-section transect (center axis)
      </p>

      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={profileData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="distance"
              stroke="#64748b"
              fontSize={10}
              tickFormatter={(v) => `${Number(v).toFixed(0)}`}
            />
            <YAxis
              stroke="#64748b"
              fontSize={10}
              tickFormatter={(v) => `${Number(v).toFixed(1)}`}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: '#e2e8f0',
                fontSize: '12px',
              }}
              formatter={(value: any) => [`${Number(value).toFixed(2)} ${unit}`, 'Elevation']}
              labelFormatter={(label) => `Distance: ${label} px`}
            />
            <Line
              type="monotone"
              dataKey="elevation"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
