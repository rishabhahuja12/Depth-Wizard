import React, { useRef, useLayoutEffect, useEffect } from 'react';
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
  verticalScale?: number;
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
 * DroneFlightController (Phase 1):
 * Places the procedural drone at a safe spawn point above terrain,
 * locks the FPV camera rigidly to the drone nose, and emits initial telemetry.
 */
export default function DroneFlightController({
  spawnPoint,
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  onTelemetryUpdate,
}: DroneFlightControllerProps) {
  const droneGroupRef = useRef<THREE.Group>(null);
  const { camera } = useThree();

  // Compute safe spawn altitude: 12m above terrain peak
  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;
  const maxElevWorld = isRelative
    ? 18.0 * verticalScale
    : (meshStats.elevation_max - meshStats.elevation_min) * verticalScale * 0.1;

  const initialY = spawnPoint ? spawnPoint[1] : Math.max(15, maxElevWorld + 10.0);
  const initialX = spawnPoint ? spawnPoint[0] : 0;
  const initialZ = spawnPoint ? spawnPoint[2] : 0;

  // Initialize drone position and camera orientation
  useLayoutEffect(() => {
    if (!droneGroupRef.current) return;
    droneGroupRef.current.position.set(initialX, initialY, initialZ);
    droneGroupRef.current.rotation.set(0, 0, 0);

    // Lock camera at the drone's front nose looking forward (-Z)
    const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
    camera.position.copy(droneGroupRef.current.position).add(noseOffset);
    camera.quaternion.copy(droneGroupRef.current.quaternion);
    camera.updateMatrixWorld();
  }, [initialX, initialY, initialZ, camera]);

  // Phase 1: Static rigid nose-lock per frame (No movement physics yet)
  useFrame(() => {
    if (!droneGroupRef.current) return;

    // Nose-locked FPV camera: rigid offset in front of the fuselage
    const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
    noseOffset.applyQuaternion(droneGroupRef.current.quaternion);
    camera.position.copy(droneGroupRef.current.position).add(noseOffset);
    camera.quaternion.copy(droneGroupRef.current.quaternion);
  });

  // Emit initial telemetry to HUD
  useEffect(() => {
    if (onTelemetryUpdate) {
      onTelemetryUpdate({
        speed: 0,
        altitudeMsl: initialY,
        altitudeAgl: initialY,
        heading: 0,
        pitch: 0,
        roll: 0,
        sensors: { left: 50, right: 50, bottom: initialY },
      });
    }
  }, [initialY, onTelemetryUpdate]);

  return (
    <group ref={droneGroupRef}>
      <DroneModel throttle={0} propRotation={0} />
      <ProximitySensors
        droneRef={droneGroupRef}
        dsmRaw={dsmRaw}
        meshStats={meshStats}
      />
    </group>
  );
}
