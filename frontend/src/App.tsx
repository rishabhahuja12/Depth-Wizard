import React, { useState } from 'react';
import { useInference } from './hooks/useInference';
import Dropzone from './components/Dropzone';
import DualViewer from './components/DualViewer';
import TerrainCanvas from './components/TerrainCanvas';
import FloodSimulator from './components/FloodSimulator';
import CrossSection from './components/CrossSection';
import SettingsPanel from './components/SettingsPanel';
import ColorBar from './components/ColorBar';
import {
  RotateCcw,
  Clock,
  Ruler,
  Globe,
  Sparkles,
  Mountain,
  Waves,
  Scissors,
  Eye,
  Download,
  Check,
} from 'lucide-react';

export default function App() {
  const { upload, data, loading, error, progress, reset } = useInference();

  // UI state
  const [verticalScale, setVerticalScale] = useState(1.5);
  const [waterLevel, setWaterLevel] = useState(0);
  const [showContours, setShowContours] = useState(false);
  const [contourInterval, setContourInterval] = useState(5);
  const [activeTab, setActiveTab] = useState<'surface' | 'flood' | 'profile' | 'inspect'>('surface');
  const [exporting, setExporting] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  // Synchronize water level and contour interval with newly loaded terrain elevation
  React.useEffect(() => {
    if (data?.calibration) {
      setWaterLevel(data.calibration.min);
      setContourInterval(data.is_georef ? 5.0 : 0.1);
    }
  }, [data]);

  const handleFileSelected = (file: File, estimateUncertainty: boolean = false) => {
    upload(file, estimateUncertainty);
  };

  const handleExport = async () => {
    if (!data?.request_id) return;
    try {
      setExporting(true);
      const response = await fetch(`/api/export/${data.request_id}`);
      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(errJson.detail || `Export failed with status ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `depthwizard_dsm_${data.request_id}.tif`;
      a.click();
      URL.revokeObjectURL(url);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 3000);
    } catch (err: any) {
      console.error('Export error:', err);
      alert(err.message || 'Export failed. Check server status.');
    } finally {
      setExporting(false);
    }
  };

  // No data yet — show Dropzone
  if (!data) {
    return (
      <div className="min-h-screen flex flex-col im-container">
        {/* Top Header */}
        <header className="border-b border-white/15 px-6 lg:px-10 py-4 flex items-center justify-between bg-[#050608]/90 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-3.5">
            <div className="w-8 h-8 border border-white/25 bg-white/5 flex items-center justify-center text-white font-mono font-bold text-xs tracking-wider">
              DW
            </div>
            <div>
              <h1 className="text-xs font-black text-white tracking-widest uppercase">
                DEPTHWIZARD
              </h1>
              <p className="text-[10px] text-neutral-400 font-mono">
                SIH26175 · ISRO SPACE APPLICATIONS CENTRE
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="im-tag">
              <span className="w-1.5 h-1.5 rounded-full bg-[#34D399]" />
              RTX 4060 READY
            </span>
            <span className="im-tag hidden sm:inline-flex">
              GAMUS STAGE 2 // LOSS 0.1500
            </span>
          </div>
        </header>

        {/* Dropzone Hero */}
        <main className="flex-1 w-full">
          <Dropzone onFileSelected={handleFileSelected} loading={loading} progress={progress} />
        </main>

        {error && (
          <div className="max-w-6xl mx-auto w-full px-6 mb-8">
            <div className="p-4 border border-rose-500/50 bg-rose-500/10 text-rose-300 font-mono text-xs flex items-center gap-2">
              <span>⚠️ ERROR:</span>
              <span>{error}</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Data loaded — show interactive 3D studio
  const cal = data.calibration;
  const stats = data.mesh_stats;

  return (
    <div className="h-screen flex flex-col im-container overflow-hidden">
      {/* Top Telemetry Header */}
      <header className="border-b border-white/15 px-6 py-3 flex items-center justify-between z-20 bg-[#050608]/95 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 border border-white/25 bg-white/5 flex items-center justify-center text-white font-mono font-bold text-xs tracking-wider">
              DW
            </div>
            <div>
              <h1 className="text-xs font-black text-white tracking-widest uppercase">
                DEPTHWIZARD
              </h1>
              <p className="text-[10px] text-neutral-400 font-mono">
                ISRO SAC · SIH26175
              </p>
            </div>
          </div>

          <div className="h-5 w-px bg-white/15 hidden md:block" />

          {/* Metric Telemetry Badges */}
          <div className="hidden sm:flex items-center gap-2">
            <span className="im-tag">
              <Clock className="w-3 h-3 text-[#38BDF8]" />
              {data.inference_time_ms.toFixed(0)}MS
            </span>
            <span className="im-tag">
              <Ruler className="w-3 h-3 text-[#34D399]" />
              {stats.original_width}×{stats.original_height}
            </span>
            <span className={`im-tag ${data.is_georef ? 'im-tag-success' : ''}`}>
              <Globe className="w-3 h-3 text-neutral-400" />
              {data.is_georef ? `${data.crs || 'EPSG:32617'} (METRIC)` : 'RELATIVE DSM'}
            </span>
            {data.confidence_mean != null && !isNaN(data.confidence_mean) && (
              <span className="im-tag">
                <Sparkles className="w-3 h-3 text-[#38BDF8]" />
                CONF {(data.confidence_mean * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExport}
            disabled={exporting}
            className="im-btn-primary"
            title="Export 32-Bit GeoTIFF raster"
          >
            {exporting ? (
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 border-2 border-current border-t-transparent animate-spin rounded-full" />
                EXPORTING...
              </span>
            ) : downloaded ? (
              <span className="flex items-center gap-2">
                <Check className="w-3.5 h-3.5" />
                DOWNLOADED GEOTIFF
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <Download className="w-3.5 h-3.5" />
                EXPORT GEOTIFF
              </span>
            )}
          </button>

          <button
            onClick={reset}
            className="im-btn-secondary"
          >
            <RotateCcw className="w-3 h-3" />
            NEW IMAGE
          </button>
        </div>
      </header>

      {/* Main Studio Viewport */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* 3D WebGL Flythrough Canvas */}
        <div className="flex-1 relative h-full">
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

          {/* Floating Elevation Bar */}
          <div className="absolute top-4 left-4 z-10">
            <ColorBar
              min={cal ? cal.min : stats.elevation_min}
              max={cal ? cal.max : stats.elevation_max}
              unit={cal ? cal.unit : 'relative'}
            />
          </div>
        </div>

        {/* Right Sidebar: Analytical Controls */}
        <aside className="w-[380px] sm:w-[420px] lg:w-[450px] xl:w-[470px] shrink-0 border-l border-white/15 bg-[#07080B] flex flex-col z-10">
          {/* Architectural Segmented Tab Navigation */}
          <div className="grid grid-cols-4 border-b border-white/15 bg-[#050608]">
            {[
              { id: 'surface' as const, num: '01', label: 'SURFACE', icon: Mountain },
              { id: 'flood' as const, num: '02', label: 'FLOOD', icon: Waves },
              { id: 'profile' as const, num: '03', label: 'PROFILE', icon: Scissors },
              { id: 'inspect' as const, num: '04', label: 'INSPECT', icon: Eye },
            ].map(({ id, num, label, icon: Icon }) => {
              const isActive = activeTab === id;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setActiveTab(id)}
                  className={`
                    py-3 px-2 flex flex-col items-center justify-center gap-1 transition-all border-r border-white/10 last:border-r-0 relative
                    ${isActive ? 'bg-white/[0.04] text-white' : 'text-neutral-400 hover:text-white hover:bg-white/[0.02]'}
                  `}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono font-bold text-neutral-400">
                      {num}
                    </span>
                    <span className="text-xs font-mono font-black tracking-wider">
                      {label}
                    </span>
                  </div>
                  {isActive && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[#38BDF8]" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Active Tab Viewport */}
          <div className="flex-1 p-6 overflow-y-auto">
            {activeTab === 'surface' && (
              <SettingsPanel
                verticalScale={verticalScale}
                onVerticalScaleChange={setVerticalScale}
                showContours={showContours}
                onContoursToggle={() => setShowContours(!showContours)}
                contourInterval={contourInterval}
                onContourIntervalChange={setContourInterval}
                meshStats={data.mesh_stats}
                calibration={data.calibration}
                isGeoref={data.is_georef}
                crs={data.crs}
              />
            )}

            {activeTab === 'flood' && (
              <FloodSimulator
                waterLevel={waterLevel}
                onWaterLevelChange={setWaterLevel}
                elevationMin={cal ? cal.min : stats.elevation_min}
                elevationMax={cal ? cal.max : stats.elevation_max}
                unit={cal ? cal.unit : 'relative'}
                dsmRaw={data.dsm_raw}
              />
            )}

            {activeTab === 'profile' && (
              <CrossSection
                dsmRaw={data.dsm_raw}
                unit={cal ? cal.unit : 'relative'}
                active={true}
                onToggle={() => {}}
                pixelSize={data.mesh_stats.pixel_size}
              />
            )}

            {activeTab === 'inspect' && (
              <DualViewer
                rgbB64={data.rgb_b64}
                dsmColorizedB64={data.dsm_colorized_b64}
                elevationMin={cal ? cal.min : stats.elevation_min}
                elevationMax={cal ? cal.max : stats.elevation_max}
                unit={cal ? cal.unit : 'relative'}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
