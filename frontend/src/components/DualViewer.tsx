import React from 'react';
import { Eye } from 'lucide-react';

interface DualViewerProps {
  rgbB64: string;
  dsmColorizedB64: string;
  elevationMin: number;
  elevationMax: number;
  unit: string;
}

export default function DualViewer({ rgbB64, dsmColorizedB64, elevationMin, elevationMax, unit }: DualViewerProps) {
  return (
    <div className="neo-card">
      <div className="neo-card-header bg-[#FFE600] text-black">
        <span className="flex items-center gap-1.5 font-black">
          <Eye className="w-4 h-4" /> 2D Dual Inspection
        </span>
        <span className="font-mono text-[10px] bg-black text-[#FFE600] px-1.5 py-0.5 border border-black">
          SYNCHRONIZED
        </span>
      </div>

      <div className="p-3 space-y-3">
        <div className="grid grid-cols-2 gap-2.5">
          {/* RGB */}
          <div className="border-2 border-black bg-black">
            <div className="bg-[#00F0FF] text-black text-[10px] font-black px-2 py-0.5 border-b-2 border-black uppercase text-center">
              RGB Optical
            </div>
            <img
              src={`data:image/jpeg;base64,${rgbB64}`}
              alt="RGB Optical"
              className="w-full aspect-square object-cover"
            />
          </div>

          {/* DSM */}
          <div className="border-2 border-black bg-black">
            <div className="bg-[#FF3366] text-white text-[10px] font-black px-2 py-0.5 border-b-2 border-black uppercase text-center">
              Turbo DSM
            </div>
            <img
              src={`data:image/png;base64,${dsmColorizedB64}`}
              alt="Estimated DSM"
              className="w-full aspect-square object-cover"
            />
          </div>
        </div>

        {/* Elevation Range Footer */}
        <div className="bg-[#0D0F12] border-2 border-black p-2 flex justify-between items-center text-[11px] font-mono font-bold">
          <span className="text-[#00FF88]">{elevationMin.toFixed(1)} {unit}</span>
          <span className="text-white/60 text-[10px] uppercase tracking-wider">Elevation Span</span>
          <span className="text-[#FF3366]">{elevationMax.toFixed(1)} {unit}</span>
        </div>
      </div>
    </div>
  );
}
