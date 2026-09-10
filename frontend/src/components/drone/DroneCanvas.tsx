import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import DroneTerrainMesh from './DroneTerrainMesh.tsx';
import DroneFlightController, { FlightTelemetry } from './DroneFlightController.tsx';
import DroneHUD from './DroneHUD.tsx';
import { SensorReadings } from './ProximitySensors.tsx';

export interface DroneCanvasProps {
  heightmapB64: string;
  rgbB64: string;
  normalMapB64?: string;
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
  };
  verticalScale: number;
  waterLevel: number;
  dsmRaw: number[][];
  spawnPoint?: [number, number, number];
  onExitFpv: () => void;
}

/**
 * DroneCanvas: Dedicated full-screen 3D FPV Drone flight viewport.
 * Renders the 3D reconstructed terrain surface and the FPV drone with nose-locked camera.
 */
export default function DroneCanvas({
  heightmapB64,
  rgbB64,
  normalMapB64,
  meshStats,
  verticalScale,
  waterLevel,
  dsmRaw,
  spawnPoint,
  onExitFpv,
}: DroneCanvasProps) {
  const [telemetry, setTelemetry] = useState<FlightTelemetry>({
    speed: 0,
    verticalSpeed: 0,
    altitudeMsl: 0,
    altitudeAgl: 0,
    heading: 0,
    pitch: 0,
    roll: 0,
    gridX: 0,
    gridZ: 0,
    sensors: { left: 50, right: 50, bottom: 25 },
  });

  // Global Esc key listener (§6)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onExitFpv();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onExitFpv]);

  return (
    <div className="w-full h-full relative select-none bg-[#050608]">
      <Canvas
        camera={{ fov: 75, near: 0.05, far: 2500 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#050608' }}
      >
        <fog attach="fog" args={['#050608', 90, 450]} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[50, 80, 50]} intensity={1.3} castShadow />
        <directionalLight position={[-30, 40, -30]} intensity={0.35} />

        {/* 3D Terrain mesh displaced from satellite DSM */}
        <DroneTerrainMesh
          heightmapB64={heightmapB64}
          rgbB64={rgbB64}
          normalMapB64={normalMapB64}
          meshStats={meshStats}
          verticalScale={verticalScale}
          waterLevel={waterLevel}
        />

        {/* Procedural drone flight rig and nose-locked FPV camera */}
        <DroneFlightController
          spawnPoint={spawnPoint}
          dsmRaw={dsmRaw}
          meshStats={meshStats}
          verticalScale={verticalScale}
          waterLevel={waterLevel}
          onTelemetryUpdate={setTelemetry}
        />

        <gridHelper args={[200, 50, '#1e293b', '#1e293b']} position={[0, -0.05, 0]} />
      </Canvas>

      {/* 2D FPV Heads-Up Display Overlay */}
      <DroneHUD
        onExitFpv={onExitFpv}
        speed={telemetry.speed}
        verticalSpeed={telemetry.verticalSpeed}
        altitudeMsl={telemetry.altitudeMsl}
        altitudeAgl={telemetry.altitudeAgl}
        heading={telemetry.heading}
        pitch={telemetry.pitch}
        roll={telemetry.roll}
        gridX={telemetry.gridX}
        gridZ={telemetry.gridZ}
        sensorLeft={telemetry.sensors.left}
        sensorRight={telemetry.sensors.right}
        sensorBottom={telemetry.sensors.bottom}
      />
    </div>
  );
}
