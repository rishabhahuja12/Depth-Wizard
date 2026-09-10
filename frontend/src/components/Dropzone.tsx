import React, { useCallback, useState, useRef } from 'react';
import { Upload, ArrowUpRight, Sparkles, Layers, Cpu, Compass, FileCheck, Grid3X3 } from 'lucide-react';
import MatrixModal from './MatrixModal';

interface DropzoneProps {
  onFileSelected: (file: File, estimateUncertainty?: boolean) => void;
  loading: boolean;
  progress: number;
}

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];

export default function Dropzone({ onFileSelected, loading, progress }: DropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);
  const [estimateUncertainty, setEstimateUncertainty] = useState(false);
  const [isMatrixOpen, setIsMatrixOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported format. Accepted: GeoTIFF (.tif, .tiff), PNG, JPG`);
      return false;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum 100 MB allowed.');
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      onFileSelected(file, estimateUncertainty);
    }
  }, [onFileSelected, estimateUncertainty]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && validateFile(file)) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      onFileSelected(file, estimateUncertainty);
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const loadSample = async (url: string, filename: string, mime: string) => {
    try {
      setLoadingSample(filename);
      setError(null);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Preset asset not found: ${filename} (HTTP ${res.status})`);
      }
      const blob = await res.blob();
      const file = new File([blob], filename, { type: mime });
      onFileSelected(file, estimateUncertainty);
    } catch (err: any) {
      setError(`Failed to load sample: ${err.message || err}`);
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-6 py-12 lg:py-20 space-y-16">
      {/* Monumental Hero Statement */}
      <div className="space-y-8">
        <h1 className="text-6xl sm:text-7xl md:text-8xl lg:text-9xl font-black tracking-tighter text-white uppercase leading-[0.85]">
          SATELLITE <br />
          <span className="stroke-text">DEPTH</span> <br />
          SYNTHESIS
        </h1>

        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pt-6 border-t border-white/15">
          <p className="text-base sm:text-lg text-neutral-400 max-w-xl font-normal leading-relaxed">
            High-precision metric Digital Surface Models (DSM) and interactive 60 FPS 3D terrain meshes synthesized from single-view optical satellite imagery.
          </p>
          <div className="font-mono text-xs text-neutral-500 uppercase tracking-widest shrink-0">
            ISRO SPACE APPLICATIONS CENTRE · SIH26175
          </div>
        </div>
      </div>

      {/* Spacious Architectural Drop Target */}
      <div
        className={`
          im-panel p-12 sm:p-16 text-center cursor-pointer transition-all border border-dashed
          ${isDragOver ? 'border-white bg-white/[0.04]' : 'border-white/20 hover:border-white/60 bg-[#07080B]'}
          ${loading ? 'pointer-events-none' : ''}
        `}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={loading ? undefined : handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          onChange={handleInputChange}
          className="hidden"
        />

        {loading ? (
          <div className="space-y-6 py-6">
            <div className="w-16 h-16 mx-auto border border-white/30 bg-white/5 flex items-center justify-center">
              <Sparkles className="w-8 h-8 text-white animate-pulse" />
            </div>

            <div className="space-y-2">
              <h3 className="text-2xl font-black tracking-tight text-white uppercase font-mono">
                Synthesizing Metric Terrain
              </h3>
              <p className="text-xs text-neutral-400 max-w-md mx-auto font-mono">
                DINOv2 ViT-S Backbone · Scale-Invariant Log1p · Two-Component DTM Decomposition
              </p>
            </div>

            <div className="w-full max-w-md mx-auto space-y-2 pt-4">
              <div className="h-1 w-full bg-white/10 overflow-hidden">
                <div
                  className="h-full bg-white transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between text-xs font-mono text-neutral-400">
                <span>INFERENCE PROGRESS</span>
                <span className="text-white font-bold">{progress}%</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6 py-4">
            <div className="w-16 h-16 mx-auto border border-white/20 bg-white/[0.02] flex items-center justify-center transition-transform duration-200 group-hover:scale-105">
              <Upload className="w-7 h-7 text-white" />
            </div>

            <div className="space-y-2">
              <h3 className="text-2xl sm:text-3xl font-black text-white tracking-tight uppercase">
                {isDragOver ? 'Release Satellite Image to Process' : 'Upload Satellite Imagery'}
              </h3>
              <p className="text-sm text-neutral-400 max-w-lg mx-auto">
                Drag and drop raw satellite GeoTIFF or high-resolution optical imagery, or click to browse local files.
              </p>
            </div>

            <div className="flex items-center justify-center gap-3 text-xs text-neutral-500 font-mono uppercase tracking-wider">
              <span>GeoTIFF (.tif)</span>
              <span>·</span>
              <span>PNG</span>
              <span>·</span>
              <span>JPEG</span>
              <span>·</span>
              <span>Max 100 MB</span>
            </div>

            {/* Uncertainty Estimation Checkbox */}
            <div className="pt-2 flex items-center justify-center">
              <label
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-3 px-5 py-2.5 border border-white/15 bg-white/[0.02] hover:border-white/40 cursor-pointer transition-colors"
              >
                <input
                  type="checkbox"
                  checked={estimateUncertainty}
                  onChange={(e) => setEstimateUncertainty(e.target.checked)}
                  className="accent-white w-4 h-4 cursor-pointer"
                />
                <span className="text-xs text-neutral-300 font-mono uppercase tracking-wider">
                  Estimate Uncertainty <span className="text-neutral-500">(MC Dropout 5-Pass)</span>
                </span>
              </label>
            </div>

            {/* Matrix Multi-Tile Grid Trigger */}
            <div className="pt-2 flex items-center justify-center">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsMatrixOpen(true);
                }}
                className="inline-flex items-center gap-2.5 px-6 py-2.5 border border-cyan-500/40 bg-cyan-500/10 hover:bg-cyan-500/20 hover:border-cyan-400 text-cyan-200 text-xs font-mono font-bold uppercase tracking-wider transition-all shadow-[0_0_20px_rgba(6,182,212,0.15)] group cursor-pointer"
              >
                <Grid3X3 className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition-transform" />
                <span>Matrix (Multi-Tile Grid)</span>
              </button>
            </div>

            {error && (
              <div className="p-4 border border-rose-500 bg-rose-500/10 text-rose-300 text-xs font-mono max-w-md mx-auto flex items-center justify-center gap-2">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Verification Benchmarks with Exaggerated Ghost Numerals */}
      <div className="space-y-8 pt-4">
        <div className="border-b border-white/15 pb-4 flex items-end justify-between">
          <div>
            <h2 className="text-xs font-mono font-bold tracking-widest text-neutral-500 uppercase">
              01 / BENCHMARKS
            </h2>
            <h3 className="text-2xl sm:text-3xl font-black text-white tracking-tight uppercase mt-1">
              Verification Testbeds
            </h3>
          </div>
          <span className="text-xs font-mono text-neutral-400 hidden sm:block">
            REAL SATELLITE TILES
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Sample 1: GeoTIFF */}
          <div
            onClick={() => loadSample('/sample_satellite_georef.tif', 'real_satellite_georef.tif', 'image/tiff')}
            className="im-panel p-8 cursor-pointer im-panel-hover flex flex-col justify-between space-y-8 relative overflow-hidden"
          >
            <div className="space-y-4">
              <span className="text-6xl sm:text-7xl font-black font-mono text-white/10 select-none block -mb-4 tracking-tighter">
                01
              </span>
              <div className="flex items-center justify-between">
                <span className="im-tag im-tag-success">EPSG:32617</span>
                <span className="im-meta">0.33m GSD</span>
              </div>
              <h4 className="text-xl font-black text-white uppercase tracking-tight">
                Multispectral Satellite GeoTIFF
              </h4>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Real urban relief tile with georeferenced coordinates and metric elevation prior calibration.
              </p>
            </div>

            <div className="flex items-center justify-between pt-6 border-t border-white/15">
              <span className="im-meta">1024×1024 · 32-BIT</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSample('/sample_satellite_georef.tif', 'real_satellite_georef.tif', 'image/tiff');
                }}
                disabled={loading || !!loadingSample}
                className="im-btn-primary"
              >
                {loadingSample === 'real_satellite_georef.tif' ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-current border-t-transparent animate-spin rounded-full" />
                    <span>LOADING...</span>
                  </>
                ) : (
                  <>
                    <span>LOAD GEOTIFF</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Sample 2: Urban PNG */}
          <div
            onClick={() => loadSample('/sample_satellite_urban.png', 'real_satellite_urban.png', 'image/png')}
            className="im-panel p-8 cursor-pointer im-panel-hover flex flex-col justify-between space-y-8 relative overflow-hidden"
          >
            <div className="space-y-4">
              <span className="text-6xl sm:text-7xl font-black font-mono text-white/10 select-none block -mb-4 tracking-tighter">
                02
              </span>
              <div className="flex items-center justify-between">
                <span className="im-tag">OPTICAL RGB</span>
                <span className="im-meta">1024×1024</span>
              </div>
              <h4 className="text-xl font-black text-white uppercase tracking-tight">
                High-Resolution Urban Optical
              </h4>
              <p className="text-xs text-neutral-400 leading-relaxed">
                Dense architectural district with high building contrast and normalized relative DSM synthesis.
              </p>
            </div>

            <div className="flex items-center justify-between pt-6 border-t border-white/15">
              <span className="im-meta">STANDARD RGB</span>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  loadSample('/sample_satellite_urban.png', 'real_satellite_urban.png', 'image/png');
                }}
                disabled={loading || !!loadingSample}
                className="im-btn-primary"
              >
                {loadingSample === 'real_satellite_urban.png' ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-current border-t-transparent animate-spin rounded-full" />
                    <span>LOADING...</span>
                  </>
                ) : (
                  <>
                    <span>LOAD PNG</span>
                    <ArrowUpRight className="w-3.5 h-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Footer System Line */}
      <div className="text-center pt-8 pb-4 border-t border-white/10">
        <p className="im-meta text-neutral-500 uppercase tracking-widest text-[10px]">
          GAMUS FINE-TUNED (0.1500 BEST LOSS) · 60 FPS WEBGL ENGINE · 32-BIT GEOTIFF PIPELINE
        </p>
      </div>

      {/* Matrix Grid Mosaic Modal */}
      <MatrixModal
        isOpen={isMatrixOpen}
        onClose={() => setIsMatrixOpen(false)}
        onSynthesize={(compositeFile) => {
          window.scrollTo({ top: 0, behavior: 'smooth' });
          onFileSelected(compositeFile, estimateUncertainty);
        }}
      />
    </div>
  );
}
