import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, AlertCircle } from 'lucide-react';

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  loading: boolean;
  progress: number;
}

const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];

export default function Dropzone({ onFileSelected, loading, progress }: DropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div className="flex items-center justify-center h-full p-8">
      <div
        className={`
          glass-panel w-full max-w-2xl p-12 text-center cursor-pointer
          transition-all duration-300 ease-out
          ${isDragOver
            ? 'border-blue-500 bg-blue-500/10 scale-[1.02] shadow-[0_0_40px_rgba(59,130,246,0.15)]'
            : 'border-dashed border-2 border-white/10 hover:border-white/20 hover:bg-white/[0.03]'
          }
        `}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={loading ? undefined : handleClick}
      >
        {loading ? (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
            <p className="text-lg text-blue-400 font-medium">Processing image...</p>
            <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-slate-400">{progress}% — Estimating depth & building terrain</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-20 h-20 mx-auto rounded-2xl bg-blue-500/10 flex items-center justify-center">
              {isDragOver ? (
                <ImageIcon className="w-10 h-10 text-blue-400" />
              ) : (
                <Upload className="w-10 h-10 text-blue-400" />
              )}
            </div>
            <div>
              <p className="text-xl font-semibold text-white">
                {isDragOver ? 'Drop image here' : 'Upload satellite imagery'}
              </p>
              <p className="text-sm text-slate-400 mt-2">
                Drag & drop or click to browse — PNG, JPG, or GeoTIFF
              </p>
            </div>
            {error && (
              <div className="flex items-center justify-center gap-2 text-red-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
