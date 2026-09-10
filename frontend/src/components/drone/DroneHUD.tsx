import React from 'react';
import { Crosshair, LogOut, Compass, Gauge, ArrowUp, Activity, AlertTriangle, Radar } from 'lucide-react';

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
 * DroneHUD:
 * 2D FPV OSD overlay with live telemetry, artificial horizon ladder,
 * heading indicator, speed gauge, 3-sensor LIDAR cluster, and terrain alerts.
 */
export default function DroneHUD({
  onExitFpv,
  speed = 0,
  altitudeMsl = 0,
  altitudeAgl = 0,
  heading = 0,
  pitch = 0,
  roll = 0,
  sensorLeft = 50,
  sensorRight = 50,
  sensorBottom = 50,
}: DroneHUDProps) {
  const speedKmh = (speed * 3.6).toFixed(1);
  const speedMs = speed.toFixed(1);

  // Compass heading direction
  const getHeadingLabel = (deg: number) => {
    const d = (deg % 360 + 360) % 360;
    if (d >= 337.5 || d < 22.5) return 'N';
    if (d >= 22.5 && d < 67.5) return 'NE';
    if (d >= 67.5 && d < 112.5) return 'E';
    if (d >= 112.5 && d < 157.5) return 'SE';
    if (d >= 157.5 && d < 202.5) return 'S';
    if (d >= 202.5 && d < 247.5) return 'SW';
    if (d >= 247.5 && d < 292.5) return 'W';
    return 'NW';
  };

  return (
    <div className="absolute inset-0 pointer-events-none z-30 select-none overflow-hidden font-mono text-white">
      {/* 1. Top Status & Exit Header */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-auto">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onExitFpv}
            className="px-3.5 py-1.5 bg-black/85 border border-white/20 hover:border-emerald-400 hover:text-emerald-400 text-white text-xs font-bold tracking-wider flex items-center gap-1.5 transition-all shadow-xl"
          >
            <LogOut className="w-3.5 h-3.5 text-emerald-400" />
            EXIT FPV [ESC]
          </button>

          <div className="px-3 py-1 bg-black/80 border border-emerald-500/40 text-emerald-400 text-[10px] font-bold tracking-widest uppercase flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            FPV DRONE ACTIVE
          </div>
        </div>

        {/* Top Center Compass Tape */}
        <div className="flex items-center gap-2 px-4 py-1.5 bg-black/85 border border-white/15 text-xs">
          <Compass className="w-3.5 h-3.5 text-[#38BDF8]" />
          <span className="text-neutral-400">HDG:</span>
          <span className="text-[#38BDF8] font-bold">{heading.toString().padStart(3, '0')}°</span>
          <span className="text-neutral-300 font-bold ml-1">({getHeadingLabel(heading)})</span>
        </div>

        {/* Right Status Badge */}
        <div className="px-3 py-1 bg-black/80 border border-white/15 text-[10px] text-neutral-400 flex items-center gap-2">
          <Activity className="w-3 h-3 text-[#38BDF8]" />
          <span>MOTORS ARMED</span>
          <span className="text-emerald-400 font-bold">100% THRUST</span>
        </div>
      </div>

      {/* 2. Artificial Horizon / Center Pitch Ladder */}
      <div
        className="absolute inset-0 flex items-center justify-center transition-transform duration-75 ease-out"
        style={{
          transform: `rotate(${-roll}deg) translateY(${pitch * 4}px)`,
        }}
      >
        {/* Pitch Ladder Bars */}
        <div className="relative w-64 h-64 flex items-center justify-center">
          {/* Horizon Line */}
          <div className="w-48 h-[1.5px] bg-emerald-400/70 shadow-[0_0_8px_rgba(52,211,153,0.6)]" />

          {/* +10 deg pitch mark */}
          <div className="absolute top-16 w-20 flex justify-between border-t border-emerald-400/40 text-[9px] text-emerald-400">
            <span>+10</span>
            <span>+10</span>
          </div>

          {/* -10 deg pitch mark */}
          <div className="absolute bottom-16 w-20 flex justify-between border-b border-emerald-400/40 text-[9px] text-emerald-400">
            <span>-10</span>
            <span>-10</span>
          </div>
        </div>
      </div>

      {/* 3. Static Center Crosshair Reticle */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative">
          <Crosshair className="w-8 h-8 text-emerald-400/80 stroke-[1.5]" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-emerald-400 rounded-full" />
        </div>
      </div>

      {/* 4. Left Flight Telemetry: Speed & Pitch */}
      <div className="absolute left-6 top-1/2 -translate-y-1/2 p-3 bg-black/85 border border-white/15 space-y-2 text-xs shadow-2xl backdrop-blur-md">
        <div className="flex items-center gap-1.5 text-neutral-400 text-[10px] uppercase font-bold tracking-wider">
          <Gauge className="w-3 h-3 text-[#38BDF8]" />
          AIRSPEED
        </div>
        <div className="text-xl font-black text-white tracking-tight">
          {speedMs} <span className="text-[10px] text-neutral-400 font-normal">M/S</span>
        </div>
        <div className="text-[10px] text-neutral-400">
          {speedKmh} <span className="text-[9px] text-neutral-500">KM/H</span>
        </div>
        <div className="pt-2 border-t border-white/10 text-[10px] space-y-0.5">
          <div className="flex justify-between gap-3">
            <span className="text-neutral-500">PITCH:</span>
            <span className="text-neutral-300 font-bold">{pitch.toFixed(1)}°</span>
          </div>
          <div className="flex justify-between gap-3">
            <span className="text-neutral-500">BANK:</span>
            <span className="text-neutral-300 font-bold">{roll.toFixed(1)}°</span>
          </div>
        </div>
      </div>

      {/* 5. Right Flight Telemetry: Altitude */}
      <div className="absolute right-6 top-1/2 -translate-y-1/2 p-3 bg-black/85 border border-white/15 space-y-2 text-xs shadow-2xl backdrop-blur-md text-right">
        <div className="flex items-center justify-end gap-1.5 text-neutral-400 text-[10px] uppercase font-bold tracking-wider">
          <ArrowUp className="w-3 h-3 text-[#34D399]" />
          ALTITUDE MSL
        </div>
        <div className="text-xl font-black text-white tracking-tight">
          {altitudeMsl.toFixed(1)} <span className="text-[10px] text-neutral-400 font-normal">M</span>
        </div>
        <div className="text-[10px] text-neutral-400">
          RADAR AGL: <span className="text-emerald-400 font-bold">{altitudeAgl.toFixed(1)} M</span>
        </div>
        <div className="pt-2 border-t border-white/10 text-[10px] text-neutral-500">
          VERTICAL DATUM: ELLIPSOID
        </div>
      </div>

      {/* 6. Bottom Controls Helper */}
      <div className="absolute bottom-4 left-6 p-2.5 bg-black/85 border border-white/15 text-[10px] text-neutral-300 flex items-center gap-3 backdrop-blur-md">
        <div className="text-emerald-400 font-bold">FPV FLIGHT:</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">W / S</kbd> Thrust</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">A / D</kbd> Yaw & Bank</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">SPACE / Q</kbd> Ascend</div>
        <div><kbd className="px-1 py-0.5 bg-white/10 text-white font-bold border border-white/20">E</kbd> Descend</div>
        <div><kbd className="px-1 py-0.5 bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">SHIFT</kbd> Turbo 2.6×</div>
      </div>

      {/* 7. 3-Sensor Proximity LIDAR Array Cluster (§5) */}
      <div className="absolute bottom-4 right-6 flex items-center gap-2 p-2.5 bg-black/85 border border-white/15 backdrop-blur-md text-xs shadow-2xl">
        <div className="flex items-center gap-1.5 text-[10px] text-neutral-400 font-bold tracking-wider px-1">
          <Radar className="w-3.5 h-3.5 text-[#38BDF8]" />
          LIDAR:
        </div>

        {/* Left Sensor */}
        <div
          className={`px-2.5 py-1 border font-bold flex items-center gap-1.5 transition-colors ${
            sensorLeft < 3
              ? 'bg-red-500/25 border-red-500 text-red-400 animate-pulse'
              : sensorLeft <= 10
              ? 'bg-amber-500/15 border-amber-500/60 text-amber-300'
              : 'bg-black/60 border-emerald-500/40 text-emerald-400'
          }`}
        >
          <span className="text-[9px] text-neutral-400">L:</span>
          <span>{sensorLeft >= 50 ? '>50m' : `${sensorLeft.toFixed(1)}m`}</span>
        </div>

        {/* Down / AGL Sensor */}
        <div
          className={`px-2.5 py-1 border font-bold flex items-center gap-1.5 transition-colors ${
            sensorBottom < 3
              ? 'bg-red-500/25 border-red-500 text-red-400 animate-pulse'
              : sensorBottom <= 10
              ? 'bg-amber-500/15 border-amber-500/60 text-amber-300'
              : 'bg-black/60 border-emerald-500/40 text-emerald-400'
          }`}
        >
          <span className="text-[9px] text-neutral-400">AGL:</span>
          <span>{sensorBottom >= 50 ? '>50m' : `${sensorBottom.toFixed(1)}m`}</span>
        </div>

        {/* Right Sensor */}
        <div
          className={`px-2.5 py-1 border font-bold flex items-center gap-1.5 transition-colors ${
            sensorRight < 3
              ? 'bg-red-500/25 border-red-500 text-red-400 animate-pulse'
              : sensorRight <= 10
              ? 'bg-amber-500/15 border-amber-500/60 text-amber-300'
              : 'bg-black/60 border-emerald-500/40 text-emerald-400'
          }`}
        >
          <span className="text-[9px] text-neutral-400">R:</span>
          <span>{sensorRight >= 50 ? '>50m' : `${sensorRight.toFixed(1)}m`}</span>
        </div>
      </div>

      {/* 8. Proximity Obstacle Warning Alert */}
      {(sensorLeft < 3 || sensorRight < 3 || sensorBottom < 2) && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 px-4 py-1.5 bg-red-500/25 border border-red-500 text-red-400 font-bold text-xs tracking-widest uppercase flex items-center gap-2 shadow-2xl animate-pulse">
          <AlertTriangle className="w-4 h-4 text-red-400" />
          <span>PROXIMITY WARNING - TERRAIN OBSTACLE CLOSE</span>
        </div>
      )}
    </div>
  );
}
