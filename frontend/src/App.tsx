import React, { useState } from 'react';
import { useInference } from './hooks/useInference';
import Dropzone from './components/Dropzone';
import DualViewer from './components/DualViewer';
import TerrainCanvas from './components/TerrainCanvas';
import FloodSimulator from './components/FloodSimulator';
import CrossSection from './components/CrossSection';
import SettingsPanel from './components/SettingsPanel';
import ColorBar from './components/ColorBar';
import { RotateCcw, Clock, Ruler } from 'lucide-react';

export default function App() {
  const { upload, data, loading, error, progress, reset } = useInference();

  // UI state
  const [verticalScale, setVerticalScale] = useState(1.5);
  const [waterLevel, setWaterLevel] = useState(0);
  const [showContours, setShowContours] = useState(false);
  const [contourInterval, setContourInterval] = useState(5);
  const [showCrossSection, setShowCrossSection] = useState(false);

  // Synchronize water level with newly loaded terrain elevation
  React.useEffect(() => {
    if (data?.calibration) {
      setWaterLevel(data.calibration.min);
    }
  }, [data]);

  const handleFileSelected = (file: File) => {
    upload(file);
  };

  // No data yet — show dropzone
  if (!data) {
    return (
      <div className="h-screen flex flex-col bg-slate-950">
        {/* Header */}
        <header className="glass-panel-strong mx-4 mt-4 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm shadow-md">
              D
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-wide">DepthWizard</h1>
              <p className="text-[10px] text-slate-400 -mt-0.5">SIH26175 · ISRO SAC</p>
            </div>
          </div>
          <span className="text-xs text-slate-400 font-medium hidden sm:inline">Single-View Height Estimation & 3D Flythrough</span>
        </header>

        {/* Dropzone */}
        <div className="flex-1">
          <Dropzone onFileSelected={handleFileSelected} loading={loading} progress={progress} />
        </div>

        {error && (
          <div className="mx-4 mb-4 glass-panel px-4 py-3 border border-red-500/20 text-red-400 text-sm flex items-center gap-2">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}
      </div>
    );
  }

  // Data loaded — show results
  const cal = data.calibration;
  const stats = data.mesh_stats;

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      {/* Header */}
      <header className="glass-panel-strong mx-2 mt-2 px-4 py-2 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-xs shadow">
            D
          </div>
          <h1 className="text-sm font-bold text-white tracking-wide">DepthWizard</h1>

          {/* Stats badges */}
          <div className="flex gap-2 ml-4">
            <span className="px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 text-[10px] font-mono border border-blue-500/20">
              <Clock className="w-3 h-3 inline mr-1" />
              {data.inference_time_ms.toFixed(0)}ms
            </span>
            <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-mono border border-emerald-500/20">
              <Ruler className="w-3 h-3 inline mr-1" />
              {stats.original_width}×{stats.original_height}
            </span>
            <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-mono border border-purple-500/20">
              {data.is_georef ? '🌐 Georeferenced' : '📷 Relative'}
            </span>
          </div>
        </div>

        <button
          onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 text-slate-300 text-xs
            hover:bg-white/10 transition-all border border-white/10 cursor-pointer"
        >
          <RotateCcw className="w-3 h-3" />
          New Image
        </button>
      </header>

      {/* Main content */}
      <div className="flex-1 flex gap-2 p-2 min-h-0">
        {/* Left sidebar */}
        <div className="w-80 flex flex-col gap-2 overflow-y-auto pr-1">
          <DualViewer
            rgbB64={data.rgb_b64}
            dsmColorizedB64={data.dsm_colorized_b64}
            elevationMin={cal.min}
            elevationMax={cal.max}
            unit={cal.unit}
          />

          <FloodSimulator
            waterLevel={waterLevel}
            onWaterLevelChange={setWaterLevel}
            elevationMin={cal.min}
            elevationMax={cal.max}
            unit={cal.unit}
            dsmRaw={data.dsm_raw}
          />

          <CrossSection
            dsmRaw={data.dsm_raw}
            unit={cal.unit}
            active={showCrossSection}
            onToggle={() => setShowCrossSection(!showCrossSection)}
            pixelSize={data.mesh_stats.pixel_size}
          />

          <SettingsPanel
            verticalScale={verticalScale}
            onVerticalScaleChange={setVerticalScale}
            showContours={showContours}
            onContoursToggle={() => setShowContours(!showContours)}
            contourInterval={contourInterval}
            onContourIntervalChange={setContourInterval}
            requestId={data.request_id}
          />
        </div>

        {/* 3D Canvas */}
        <div className="flex-1 glass-panel overflow-hidden relative">
          <TerrainCanvas
            heightmapB64={data.heightmap_b64}
            rgbB64={data.rgb_b64}
            normalMapB64={data.normal_map_b64}
            meshStats={data.mesh_stats}
            verticalScale={verticalScale}
            waterLevel={waterLevel}
            showContours={showContours}
            contourInterval={contourInterval}
            dsmRaw={data.dsm_raw}
          />
          <div className="absolute top-3 right-3 pointer-events-none">
            <ColorBar min={cal.min} max={cal.max} unit={cal.unit} />
          </div>
        </div>
      </div>
    </div>
  );
}
