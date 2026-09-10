import React from 'react';
import { Crosshair, LogOut } from 'lucide-react';

export interface DroneHUDProps {
  onExitFpv: () => void;
  speed?: number;
  altitudeMsl?: number;
  altitudeAgl?: number;
  heading?: number;
  pitch?: number;
  roll?: number;
  sensorLeft?: number;
  sensorRight?: number;
  sensorBottom?: number;
  coordinates?: string;
}

/**
 * DroneHUD: FPV OSD overlay scaffold.
 */
export default function DroneHUD({ onExitFpv }: DroneHUDProps) {
  return (
    <div className="absolute inset-0 pointer-events-none z-30 select-none overflow-hidden font-mono">
      {/* Top Status Bar */}
      <div className="absolute top-4 left-4 flex items-center gap-2 pointer-events-auto">
        <button
          type="button"
          onClick={onExitFpv}
          className="px-3 py-1.5 bg-black/80 border border-white/20 hover:border-white/50 text-white text-xs font-bold tracking-wider flex items-center gap-1.5 transition-all"
        >
          <LogOut className="w-3.5 h-3.5 text-[#38BDF8]" />
          EXIT FPV MODE [ESC]
        </button>

        <div className="px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold tracking-widest uppercase flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 bg-emerald-400 animate-ping" />
          FPV DRONE RECONNAISSANCE ACTIVE
        </div>
      </div>

      {/* Center Reticle Placeholder */}
      <div className="absolute inset-0 flex items-center justify-center opacity-40">
        <Crosshair className="w-10 h-10 text-white stroke-[1]" />
      </div>

      {/* Bottom Telemetry Legend */}
      <div className="absolute bottom-4 left-4 p-2.5 bg-black/85 border border-white/15 text-[10px] text-neutral-400 space-y-1">
        <div className="text-white font-bold">FPV FLIGHT CONTROLS</div>
        <div className="flex gap-2">
          <span className="text-neutral-300">W/S</span> Pitch Forward/Back
          <span className="text-neutral-300 ml-2">A/D</span> Yaw Left/Right
          <span className="text-neutral-300 ml-2">SPACE/Q</span> Up
          <span className="text-neutral-300 ml-2">E</span> Down
          <span className="text-neutral-300 ml-2">SHIFT</span> Boost
        </div>
      </div>
    </div>
  );
}
