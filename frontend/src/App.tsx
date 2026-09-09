import React, { useState } from 'react';
import { useInference } from './hooks/useInference';
import Dropzone from './components/Dropzone';
import DualViewer from './components/DualViewer';
import TerrainCanvas from './components/TerrainCanvas';
import FloodSimulator from './components/FloodSimulator';
import CrossSection from './components/CrossSection';
import SettingsPanel from './components/SettingsPanel';
import ColorBar from './components/ColorBar';
import { RotateCcw, Clock, Ruler, Globe, Sparkles } from 'lucide-react';

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

  // No data yet — show Neo-Brutalist Dropzone
  if (!data) {
    return (
      <div className="h-screen flex flex-col bg-[#0A0C10]">
        {/* Neo-Brutalist Header */}
        <header className="bg-[#161A22] border-b-4 border-black px-6 py-3 flex items-center justify-between shadow-[0_4px_0px_0px_#000]">
          <div className="flex items-center gap-3">
            <div className="bg-[#FFE600] text-black font-black text-base px-2.5 py-1 border-2 border-black shadow-[3px_3px_0px_0px_#000]">
              DW
            </div>
            <div>
              <h1 className="text-xl font-black text-white tracking-wider uppercase">
                DEPTHWIZARD <span className="text-[#FFE600]">// 3D</span>
              </h1>
              <p className="text-[10px] font-mono font-bold text-[#00F0FF] uppercase -mt-0.5">
                SIH26175 · ISRO SPACE APPLICATIONS CENTRE
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="neo-badge bg-[#00F0FF] text-black hidden sm:inline-flex">
              SINGLE-VIEW DSM ESTIMATION
            </span>
            <span className="neo-badge bg-[#00FF88] text-black">
              RTX 4060 ACCELERATED
            </span>
          </div>
        </header>

        {/* Dropzone Hero */}
        <div className="flex-1 overflow-y-auto">
          <Dropzone onFileSelected={handleFileSelected} loading={loading} progress={progress} />
        </div>

        {error && (
          <div className="mx-6 mb-6 p-4 bg-[#FF3366] text-white border-3 border-black shadow-[4px_4px_0px_0px_#000] font-bold text-sm flex items-center gap-2">
            <span>⚠️ ERROR:</span>
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
    <div className="h-screen flex flex-col bg-[#0A0C10]">
      {/* Neo-Brutalist Top Bar */}
      <header className="bg-[#161A22] border-b-4 border-black px-4 py-2.5 flex items-center justify-between z-20 shadow-[0_4px_0px_0px_#000]">
        <div className="flex items-center gap-3">
          <div className="bg-[#FFE600] text-black font-black text-sm px-2 py-0.5 border-2 border-black shadow-[2px_2px_0px_0px_#000]">
            DW
          </div>
          <h1 className="text-sm font-black text-white tracking-wider uppercase">
            DEPTHWIZARD
          </h1>

          {/* Neo-Brutalist Badges */}
          <div className="flex items-center gap-2 ml-3">
            <span className="neo-badge bg-[#00F0FF] text-black">
              <Clock className="w-3 h-3" />
              {data.inference_time_ms.toFixed(0)}MS
            </span>
            <span className="neo-badge bg-[#00FF88] text-black">
              <Ruler className="w-3 h-3" />
              {stats.original_width}×{stats.original_height}
            </span>
            <span className={`neo-badge ${data.is_georef ? 'bg-[#FFE600] text-black' : 'bg-white text-black'}`}>
              <Globe className="w-3 h-3" />
              {data.is_georef ? 'GEOREFERENCED' : 'RELATIVE DSM'}
            </span>
            {data.confidence_mean !== undefined && (
              <span className="neo-badge bg-[#A78BFA] text-black">
                <Sparkles className="w-3 h-3" />
                CONF {(data.confidence_mean * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>

        <button
          onClick={reset}
          className="neo-btn bg-[#FF3366] text-white px-3.5 py-1.5 text-xs font-black hover:bg-[#ff4d79]"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          NEW IMAGE
        </button>
      </header>

      {/* Main Studio Viewport */}
      <div className="flex-1 flex gap-3 p-3 min-h-0 bg-[#0A0C10]">
        {/* Left Analytical Sidebar */}
        <div className="w-[370px] flex flex-col gap-3 overflow-y-auto pr-1">
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

        {/* 3D WebGL Viewport Container */}
        <div className="flex-1 bg-[#10131A] border-4 border-black shadow-[6px_6px_0px_0px_#000000] overflow-hidden relative">
          {/* Viewport Header Strip */}
          <div className="absolute top-3 left-3 z-10 pointer-events-none flex items-center gap-2">
            <span className="neo-badge bg-black text-[#FFE600] border-2 border-black shadow-[2px_2px_0px_0px_#000]">
              INTERACTIVE 3D FLYTHROUGH
            </span>
            <span className="neo-badge bg-[#00FF88] text-black">
              LOCKED 60 FPS
            </span>
          </div>

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

          {/* Floating Neo-Brutalist Colormap Legend */}
          <div className="absolute top-3 right-3 z-10 pointer-events-auto">
            <ColorBar min={cal.min} max={cal.max} unit={cal.unit} />
          </div>
        </div>
      </div>
    </div>
  );
}
