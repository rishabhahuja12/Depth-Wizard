import React, { useState } from 'react';
import { useInference } from './hooks/useInference';
import Dropzone from './components/Dropzone';
import DualViewer from './components/DualViewer';
import TerrainCanvas from './components/TerrainCanvas';
import VoxelTerrain from './components/VoxelTerrain';
import Minimap from './components/Minimap';
import { useFlightMode } from './hooks/useFlightMode';
import DroneCanvas from './components/drone/DroneCanvas';
import { Radio } from 'lucide-react';
import FloodSimulator from './components/FloodSimulator';
import CrossSection from './components/CrossSection';
import SettingsPanel from './components/SettingsPanel';
import TerrainQuickPanel from './components/TerrainQuickPanel';
import ColorBar from './components/ColorBar';
import { useTerrainSettings } from './hooks/useTerrainSettings';
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
  Sliders,
} from 'lucide-react';

export default function App() {
  const { upload, data, loading, error, progress, reset } = useInference();
  const { isFpv, toggleFlightMode, exitFpv } = useFlightMode('studio');

  // Unified Terrain Settings (§5)
  const terrainSettings = useTerrainSettings({
    initialVerticalScale: 1.5,
    initialWaterLevel: 0,
    initialShowContours: false,
    initialContourInterval: 5,
    initialRenderMode: 'voxel',
    initialVoxelBands: 8,
    initialVoxelResolution: 64,
  });

  const {
    verticalScale,
    setVerticalScale,
    waterLevel,
    setWaterLevel,
    showContours,
    setShowContours,
    contourInterval,
    setContourInterval,
    renderMode,
    setRenderMode,
    voxelBands,
    setVoxelBands,
    voxelResolution,
    setVoxelResolution,
  } = terrainSettings;

  const [isQuickPanelOpen, setIsQuickPanelOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'surface' | 'flood' | 'profile' | 'inspect'>('surface');
  const [droneSpawnPreset, setDroneSpawnPreset] = useState<'center' | 'north' | 'south' | 'west' | 'east'>('center');
  const [exporting, setExporting] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  // Compute 3D spawn coordinates based on selected reconnaissance ingress preset (§3 & §6)
  const droneSpawnPoint = React.useMemo<[number, number, number] | undefined>(() => {
    if (!data?.mesh_stats) return undefined;
    const w = data.mesh_stats.width * 0.1;
    const d = data.mesh_stats.height * 0.1;
    switch (droneSpawnPreset) {
      case 'north':
        return [0, 0, -d * 0.38];
      case 'south':
        return [0, 0, d * 0.38];
      case 'west':
        return [-w * 0.38, 0, 0];
      case 'east':
        return [w * 0.38, 0, 0];
      case 'center':
      default:
        return [0, 0, 0];
    }
  }, [data?.mesh_stats, droneSpawnPreset]);

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

          {/* Terrain & Voxel Detail Quick Panel Trigger (§5) */}
          <button
            type="button"
            onClick={() => setIsQuickPanelOpen((prev) => !prev)}
            className={`py-2 px-3 text-xs font-mono font-bold border transition-all flex items-center gap-1.5 ${
              isQuickPanelOpen
                ? 'bg-[#38BDF8] text-black border-[#38BDF8] shadow-lg shadow-[#38BDF8]/20'
                : 'bg-transparent text-neutral-300 border-white/20 hover:border-white/50 hover:text-white hover:bg-white/5'
            }`}
            title="Toggle Terrain & Voxel Detail Quick Panel [Key P]"
          >
            <Sliders className="w-3.5 h-3.5 text-[#38BDF8]" />
            TERRAIN [P]
          </button>

          <button
            type="button"
            onClick={() => {
              if (document.pointerLockElement) {
                document.exitPointerLock();
              }
              toggleFlightMode();
            }}
            className={`py-2 px-3 text-xs font-mono font-bold border transition-all flex items-center gap-1.5 ${
              isFpv
                ? 'bg-[#10B981] text-black border-[#10B981] shadow-lg'
                : 'bg-transparent text-white border-white/20 hover:border-white/50 hover:bg-white/5'
            }`}
            title="Toggle First-Person View Drone Flight Mode"
          >
            <Radio className="w-3.5 h-3.5 text-[#38BDF8]" />
            {isFpv ? 'EXIT FPV' : 'DRONE FPV'}
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
        {isFpv ? (
          <DroneCanvas
            heightmapB64={data.heightmap_b64}
            rgbB64={data.rgb_b64}
            normalMapB64={data.normal_map_b64}
            meshStats={data.mesh_stats}
            verticalScale={verticalScale}
            waterLevel={waterLevel}
            dsmRaw={data.dsm_raw}
            spawnPoint={droneSpawnPoint}
            renderMode={renderMode}
            voxelResolution={voxelResolution}
            voxelBands={voxelBands}
            onOpenQuickPanel={() => setIsQuickPanelOpen((prev) => !prev)}
            onExitFpv={() => {
              if (document.pointerLockElement) {
                document.exitPointerLock();
              }
              exitFpv();
            }}
          />
        ) : (
          <>
            {/* 3D WebGL Main Viewport */}
            <div className="flex-1 relative h-full">
          {renderMode === 'voxel' ? (
            <VoxelTerrain
              dsmRaw={data.dsm_raw}
              meshStats={data.mesh_stats}
              verticalScale={verticalScale}
              waterLevel={waterLevel}
              targetResolution={voxelResolution}
              bandCount={voxelBands}
            />
          ) : (
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
          )}

          {/* Smooth Mesh Minimap Overlay (§3.6) */}
          {renderMode === 'voxel' && (
            <div className="absolute top-4 right-4 z-20">
              <Minimap
                heightmapB64={data.heightmap_b64}
                rgbB64={data.rgb_b64}
                normalMapB64={data.normal_map_b64}
                meshStats={data.mesh_stats}
                verticalScale={verticalScale}
                waterLevel={waterLevel}
                showContours={false}
                dsmRaw={data.dsm_raw}
              />
            </div>
          )}

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
                renderMode={renderMode}
                onRenderModeChange={setRenderMode}
                voxelBands={voxelBands}
                onVoxelBandsChange={setVoxelBands}
                voxelResolution={voxelResolution}
                onVoxelResolutionChange={setVoxelResolution}
                meshStats={data.mesh_stats}
                calibration={data.calibration}
                isGeoref={data.is_georef}
                crs={data.crs}
                droneSpawnPreset={droneSpawnPreset}
                onDroneSpawnPresetChange={setDroneSpawnPreset}
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
          </>
        )}
      </div>

      {/* Floating Terrain & Voxel Detail Quick Panel Overlay (§5) */}
      <TerrainQuickPanel
        isOpen={isQuickPanelOpen}
        onClose={() => setIsQuickPanelOpen(false)}
        settings={terrainSettings}
      />
    </div>
  );
}
