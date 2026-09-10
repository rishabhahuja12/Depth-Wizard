import React, { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import DroneModel from './DroneModel.tsx';
import ProximitySensors, { SensorReadings } from './ProximitySensors.tsx';

export interface DroneFlightControllerProps {
  spawnPoint?: [number, number, number];
  dsmRaw: number[][];
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
  };
  onTelemetryUpdate?: (telemetry: {
    speed: number;
    altitudeMsl: number;
    altitudeAgl: number;
    heading: number;
    pitch: number;
    roll: number;
    sensors: SensorReadings;
  }) => void;
}

/**
 * DroneFlightController scaffold.
 */
export default function DroneFlightController({
  spawnPoint = [0, 25, 0],
  dsmRaw,
  meshStats,
}: DroneFlightControllerProps) {
  const droneGroupRef = useRef<THREE.Group>(null);
  const { camera } = useThree();

  // Initial camera position relative to drone
  useFrame(() => {
    if (!droneGroupRef.current) return;
    // Set initial spawn position if needed
    if (droneGroupRef.current.position.lengthSq() === 0) {
      droneGroupRef.current.position.set(...spawnPoint);
    }
  });

  return (
    <group ref={droneGroupRef}>
      <DroneModel throttle={0} />
      <ProximitySensors
        droneRef={droneGroupRef}
        dsmRaw={dsmRaw}
        meshStats={meshStats}
      />
    </group>
  );
}
