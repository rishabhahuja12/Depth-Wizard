import React from 'react';
import { Settings, Mountain, Download } from 'lucide-react';
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

  const handleExport = async () => {
    try {
      const response = await fetch(`/api/export/${requestId}`);
      if (!response.ok) throw new Error('Export failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `depthwizard_dsm_${requestId}.tif`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export error:', err);
    }
  };

  return (
    <div className="glass-panel p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Settings className="w-4 h-4 text-blue-400" />
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Settings</h3>
      </div>

      {/* Vertical Exaggeration */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Mountain className="w-3 h-3" />
          <span>Vertical Scale</span>
          <span className="ml-auto font-mono text-blue-400">{verticalScale.toFixed(1)}×</span>
        </div>
        <input
          type="range"
          min={0.1}
          max={5}
          step={0.1}
          value={verticalScale}
          onChange={(e) => onVerticalScaleChange(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Contour Lines */}
      <ContourOverlay
        active={showContours}
        onToggle={onContoursToggle}
        interval={contourInterval}
        onIntervalChange={onContourIntervalChange}
      />

      {/* Export */}
      <button
        onClick={handleExport}
        className="flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm w-full
          bg-emerald-500/10 text-emerald-400 border border-emerald-500/20
          hover:bg-emerald-500/20 transition-all font-medium cursor-pointer"
      >
        <Download className="w-4 h-4" />
        Export GeoTIFF
      </button>
    </div>
  );
}
