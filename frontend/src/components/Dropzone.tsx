import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, AlertCircle, Sparkles, MapPin } from 'lucide-react';

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  loading: boolean;
  progress: number;
}

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];

export default function Dropzone({ onFileSelected, loading, progress }: DropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported format. Accepted: PNG, JPG, TIFF`);
      return false;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum 100 MB.');
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
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = ALLOWED_EXTENSIONS.join(',');
    input.onchange = (e: any) => {
      const file = e.target.files[0];
      if (file && validateFile(file)) {
        onFileSelected(file);
      }
    };
    input.click();
  };

  const loadSample = async (url: string, filename: string, mime: string) => {
    try {
      setLoadingSample(filename);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: mime });
      onFileSelected(file);
    } catch (err) {
      setError(`Failed to load sample: ${err}`);
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-full max-w-4xl mx-auto px-4 py-8">
      {/* Neo-Brutalist Main Upload Card */}
      <div className="w-full bg-[#1A1D24] border-4 border-black shadow-[10px_10px_0px_0px_#000000]">
        {/* Card Header */}
        <div className="bg-[#FFE600] text-black border-b-4 border-black px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-3.5 h-3.5 bg-black inline-block" />
            <span className="font-black text-sm tracking-wider uppercase">Satellite Image Ingestion Terminal</span>
          </div>
          <span className="font-mono text-xs font-black bg-black text-[#FFE600] px-2 py-0.5 border border-black">
            SIH26175 // SAC
          </span>
        </div>

        {/* Drop Area */}
        <div
          className={`
            p-10 text-center cursor-pointer transition-all duration-150
            ${isDragOver ? 'bg-[#252B3B] border-4 border-dashed border-[#00F0FF]' : 'hover:bg-[#202531]'}
            ${loading ? 'pointer-events-none opacity-90' : ''}
          `}
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={loading ? undefined : handleClick}
        >
          {loading ? (
            <div className="space-y-6 py-6">
              <div className="w-20 h-20 mx-auto bg-[#00F0FF] border-4 border-black shadow-[6px_6px_0px_0px_#000] flex items-center justify-center animate-bounce">
                <Sparkles className="w-10 h-10 text-black" />
              </div>
              <div>
                <h3 className="text-2xl font-black text-white tracking-wide uppercase">
                  Processing Satellite Image...
                </h3>
                <p className="text-sm font-mono text-[#00F0FF] mt-1">
                  Predicting elevation · Computing normal vectors · Synthesizing 3D mesh
                </p>
              </div>
              <div className="w-full max-w-md mx-auto bg-black border-2 border-white/20 p-1">
                <div
                  className="h-5 bg-[#FFE600] border-2 border-black transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="font-mono text-xs text-white/60">{progress}% COMPLETE</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="w-24 h-24 mx-auto bg-[#FFE600] border-4 border-black shadow-[6px_6px_0px_0px_#000] flex items-center justify-center transform hover:rotate-3 transition-transform">
                {isDragOver ? (
                  <ImageIcon className="w-12 h-12 text-black" />
                ) : (
                  <Upload className="w-12 h-12 text-black" />
                )}
              </div>

              <div>
                <h2 className="text-3xl font-black text-white uppercase tracking-tight">
                  {isDragOver ? 'Drop Satellite Tile Here' : 'Drag & Drop Satellite Image'}
                </h2>
                <p className="text-sm font-mono text-[#00F0FF] mt-2 font-bold">
                  OR CLICK TO BROWSE LOCAL FILES
                </p>
                <div className="flex items-center justify-center gap-3 mt-4">
                  <span className="neo-badge bg-[#00FF88] text-black">TIFF / GeoTIFF</span>
                  <span className="neo-badge bg-[#FFE600] text-black">PNG</span>
                  <span className="neo-badge bg-[#FF3366] text-white">JPG / JPEG</span>
                  <span className="neo-badge bg-[#00F0FF] text-black">Max 100 MB</span>
                </div>
              </div>

              {error && (
                <div className="p-3 bg-[#FF3366] text-white border-2 border-black font-bold text-sm shadow-[4px_4px_0px_0px_#000] flex items-center justify-center gap-2">
                  <AlertCircle className="w-5 h-5" />
                  {error}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Quick Sample Loader Bar */}
        <div className="border-t-4 border-black bg-[#12151B] p-5">
          <div className="flex items-center justify-between mb-3">
            <span className="font-black text-xs text-[#FFE600] uppercase tracking-wider flex items-center gap-2">
              <MapPin className="w-4 h-4" /> Quick-Test High-Res Samples:
            </span>
            <span className="font-mono text-[11px] text-white/50">One-click live demo</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              type="button"
              disabled={loading || !!loadingSample}
              onClick={(e) => {
                e.stopPropagation();
                loadSample('/sample_satellite_georef.tif', 'real_satellite_georef.tif', 'image/tiff');
              }}
              className="neo-btn bg-[#00FF88] text-black p-3 text-left flex items-center justify-between hover:bg-[#20ff98]"
            >
              <div>
                <div className="font-black text-xs">SAMPLE 1: Real GeoTIFF (0.33m)</div>
                <div className="font-mono text-[10px] text-black/70 mt-0.5">UTM 17N (EPSG:32617) · Urban relief</div>
              </div>
              <span className="bg-black text-[#00FF88] text-[10px] font-black px-2 py-1 border border-black">
                {loadingSample === 'real_satellite_georef.tif' ? 'LOADING...' : 'LOAD TIF'}
              </span>
            </button>

            <button
              type="button"
              disabled={loading || !!loadingSample}
              onClick={(e) => {
                e.stopPropagation();
                loadSample('/sample_satellite_urban.png', 'real_satellite_urban.png', 'image/png');
              }}
              className="neo-btn bg-[#00F0FF] text-black p-3 text-left flex items-center justify-between hover:bg-[#3bf4ff]"
            >
              <div>
                <div className="font-black text-xs">SAMPLE 2: Urban Optical PNG</div>
                <div className="font-mono text-[10px] text-black/70 mt-0.5">1024×1024 RGB · Relative DSM mode</div>
              </div>
              <span className="bg-black text-[#00F0FF] text-[10px] font-black px-2 py-1 border border-black">
                {loadingSample === 'real_satellite_urban.png' ? 'LOADING...' : 'LOAD PNG'}
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
