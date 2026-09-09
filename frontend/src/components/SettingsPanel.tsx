import React, { useState } from 'react';
import { Settings, Mountain, Download, Check } from 'lucide-react';
import ContourOverlay from './ContourOverlay';

interface SettingsPanelProps {
  verticalScale: number;
  onVerticalScaleChange: (val: number) => void;
  showContours: boolean;
  onContoursToggle: () => void;
  contourInterval: number;
  onContourIntervalChange: (val: number) => void;
  requestId: string;
}

export default function SettingsPanel({
  verticalScale, onVerticalScaleChange,
  showContours, onContoursToggle,
  contourInterval, onContourIntervalChange,
  requestId,
}: SettingsPanelProps) {
  const [exporting, setExporting] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  const handleExport = async () => {
    try {
      setExporting(true);
      const response = await fetch(`/api/export/${requestId}`);
      if (!response.ok) throw new Error('Export failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `depthwizard_dsm_${requestId}.tif`;
      a.click();
      URL.revokeObjectURL(url);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 3000);
    } catch (err) {
      console.error('Export error:', err);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="neo-card">
      <div className="neo-card-header bg-[#00FF88] text-black">
        <span className="flex items-center gap-1.5 font-black">
          <Settings className="w-4 h-4" /> Terrain Controls
        </span>
        <span className="font-mono text-[10px] bg-black text-[#00FF88] px-1.5 py-0.5 border border-black">
          OUTPUT
        </span>
      </div>

      <div className="p-3.5 space-y-4">
        {/* Vertical Exaggeration */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs font-mono font-bold">
            <span className="flex items-center gap-1.5 text-white/80 uppercase">
              <Mountain className="w-3.5 h-3.5 text-[#00FF88]" /> Vertical Exaggeration
            </span>
            <span className="bg-[#FFE600] text-black px-1.5 py-0.2 border border-black font-black">
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
            className="neo-slider"
          />
          <div className="flex justify-between text-[9px] font-mono text-white/40">
            <span>0.1× (Subtle)</span>
            <span>5.0× (Extreme)</span>
          </div>
        </div>

        {/* Contour Lines */}
        <ContourOverlay
          active={showContours}
          onToggle={onContoursToggle}
          interval={contourInterval}
          onIntervalChange={onContourIntervalChange}
        />

        {/* Export GeoTIFF Button */}
        <button
          onClick={handleExport}
          disabled={exporting}
          className="w-full neo-btn bg-[#FFE600] text-black py-2.5 px-4 font-black text-xs uppercase tracking-wider hover:bg-[#ffeb3b] active:translate-x-1 active:translate-y-1"
        >
          {exporting ? (
            <span className="flex items-center gap-2">
              <div className="w-3 h-3 border-2 border-black border-t-transparent animate-spin" />
              Generating GeoTIFF...
            </span>
          ) : downloaded ? (
            <span className="flex items-center gap-2 text-black">
              <Check className="w-4 h-4" /> Downloaded Successfully
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <Download className="w-4 h-4" /> Export Standard GeoTIFF
            </span>
          )}
        </button>
      </div>
    </div>
  );
}
