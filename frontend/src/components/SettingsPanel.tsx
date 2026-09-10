import React from 'react';
import { Mountain, Layers, Compass } from 'lucide-react';
import ContourOverlay from './ContourOverlay';

interface MeshStats {
  elevation_min: number;
  elevation_max: number;
  original_width: number;
  original_height: number;
  pixel_size?: number;
}

interface Calibration {
  min: number;
  max: number;
  unit: string;
}

interface SettingsPanelProps {
  verticalScale: number;
  onVerticalScaleChange: (val: number) => void;
  showContours: boolean;
  onContoursToggle: () => void;
  contourInterval: number;
  onContourIntervalChange: (val: number) => void;
  meshStats?: MeshStats;
  calibration?: Calibration | null;
  isGeoref?: boolean;
  crs?: string;
}

export default function SettingsPanel({
  verticalScale,
  onVerticalScaleChange,
  showContours,
  onContoursToggle,
  contourInterval,
  onContourIntervalChange,
  meshStats,
  calibration,
  isGeoref = false,
  crs,
}: SettingsPanelProps) {
  const minElev = calibration ? calibration.min : (meshStats?.elevation_min ?? 0);
  const maxElev = calibration ? calibration.max : (meshStats?.elevation_max ?? 1);
  const spanElev = Math.max(0, maxElev - minElev);
  const unit = calibration ? calibration.unit : 'relative';
  const gsd = meshStats?.pixel_size ? `${meshStats.pixel_size.toFixed(2)}m` : 'SYNTHETIC';

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2 border-b border-white/15 pb-4">
        <div className="text-[10px] font-mono font-bold tracking-widest text-neutral-500 uppercase">
          01 / SURFACE CONTROLS
        </div>
        <h3 className="text-lg font-black text-white uppercase tracking-tight">
          Relief & Geometry
        </h3>
        <p className="text-xs text-neutral-400 leading-relaxed font-normal">
          Adjust vertical scale exaggeration, realtime vector contour lines, and inspect geometric telemetry.
        </p>
      </div>

      <div className="space-y-7">
        {/* Vertical Exaggeration */}
        <div className="space-y-3.5">
          <div className="flex justify-between items-end text-xs font-mono">
            <span className="text-neutral-400 uppercase tracking-wider text-[11px]">
              Vertical Scale
            </span>
            <span className="text-white font-black text-lg tracking-tight">
              {verticalScale.toFixed(1)}×
            </span>
          </div>

          <input
            type="range"
            min={0.1}
            max={5}
            step={0.1}
            value={verticalScale}
            onChange={(e) => onVerticalScaleChange(parseFloat(e.target.value))}
            className="im-slider"
          />

          <div className="flex justify-between text-[10px] text-neutral-500 font-mono">
            <span>0.1× SUBTLE</span>
            <span>5.0× STEEP</span>
          </div>

          {/* Scale Presets - Sharp Minimalist Grid */}
          <div className="grid grid-cols-4 gap-2 pt-1">
            {[
              { label: '0.5×', val: 0.5 },
              { label: '1.0×', val: 1.0 },
              { label: '1.8×', val: 1.8 },
              { label: '3.0×', val: 3.0 },
            ].map(({ label, val }) => {
              const isSelected = Math.abs(verticalScale - val) < 0.05;
              return (
                <button
                  key={label}
                  type="button"
                  onClick={() => onVerticalScaleChange(val)}
                  className={`py-2 text-xs font-mono font-bold border transition-all ${
                    isSelected
                      ? 'bg-white text-black border-white'
                      : 'bg-transparent text-neutral-400 border-white/15 hover:border-white/40 hover:text-white'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Contour Lines Section */}
        <div className="pt-5 border-t border-white/15">
          <ContourOverlay
            active={showContours}
            onToggle={onContoursToggle}
            interval={contourInterval}
            onIntervalChange={onContourIntervalChange}
            isGeoref={isGeoref}
          />
        </div>

        {/* Elevation & Mesh Telemetry Card */}
        <div className="pt-5 border-t border-white/15 space-y-3">
          <div className="text-[10px] font-mono font-bold uppercase tracking-widest text-neutral-500">
            Elevation & Spatial Telemetry
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Base Elevation</span>
              <span className="text-white font-bold text-sm mt-1 block">{minElev.toFixed(1)} {unit}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Peak Elevation</span>
              <span className="text-[#38BDF8] font-bold text-sm mt-1 block">{maxElev.toFixed(1)} {unit}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Relief Span</span>
              <span className="text-white font-bold text-sm mt-1 block">{spanElev.toFixed(1)} {unit}</span>
            </div>
            <div className="p-3 bg-white/[0.02] border border-white/10">
              <span className="text-[10px] text-neutral-400 block uppercase tracking-wider">Pixel GSD</span>
              <span className="text-[#34D399] font-bold text-sm mt-1 block">{gsd}</span>
            </div>
          </div>
        </div>

        {/* Coordinate Reference Box */}
        <div className="p-3.5 bg-white/[0.02] border border-white/10 text-xs font-mono space-y-2">
          <div className="flex items-center justify-between gap-3">
            <span className="text-neutral-400 shrink-0 uppercase tracking-wider text-[10px]">Projection</span>
            <span className="text-white font-bold text-right truncate text-[11px]">{isGeoref ? `${crs || 'EPSG:32617'} (WGS84 UTM)` : 'RELATIVE SYNTHETIC GRID'}</span>
          </div>
          <div className="flex items-center justify-between gap-3 pt-2 border-t border-white/10">
            <span className="text-neutral-400 shrink-0 uppercase tracking-wider text-[10px]">Datum</span>
            <span className="text-white font-bold text-right truncate text-[11px]">{isGeoref ? 'ELLIPSOIDAL HEIGHT (M)' : 'NORMALIZED RANGE [0, 1]'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
