import React, { useRef, useLayoutEffect, useState } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import DroneModel from './DroneModel.tsx';
import ProximitySensors, { SensorReadings } from './ProximitySensors.tsx';
import { useDroneInput } from './useDroneInput.ts';
import {
  getTerrainElevationAt,
  computeAgl,
  computeMsl,
  resolveTerrainCollision,
} from './dsmSampling.ts';

export interface FlightTelemetry {
  speed: number;
  verticalSpeed: number;
  altitudeMsl: number;
  altitudeAgl: number;
  heading: number;
  pitch: number;
  roll: number;
  gridX: number;
  gridZ: number;
  sensors: SensorReadings;
  autopilotMode: 'manual' | 'orbit' | 'transect';
  cameraGimbal: boolean;
}

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
  waterLevel?: number;
  autopilotMode?: 'manual' | 'orbit' | 'transect';
  cameraGimbal?: boolean;
  onAutopilotModeChange?: (mode: 'manual' | 'orbit' | 'transect') => void;
  onCameraGimbalToggle?: () => void;
  onTelemetryUpdate?: (telemetry: FlightTelemetry) => void;
}

/**
 * DroneFlightController (Phase 2 - 7):
 * - Newtonian flight physics with velocity integration, drag, auto-bank roll, thrust-pitch coupling
 * - Phase 3 AGL clearance & hard-floor/rooftop collision resolution
 * - Phase 4 Proximity sensor soft-bump velocity zeroing
 * - Phase 7 Autopilot Orbital POI recon, Terrain-Hugging Transect cruise, and Horizon-Stabilized Gimbal
 */
export default function DroneFlightController({
  spawnPoint,
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel,
  autopilotMode: controlledMode,
  cameraGimbal: controlledGimbal,
  onAutopilotModeChange,
  onCameraGimbalToggle,
  onTelemetryUpdate,
}: DroneFlightControllerProps) {
  const droneGroupRef = useRef<THREE.Group>(null);
  const { camera } = useThree();

  // Internal autopilot state with external control sync
  const [internalMode, setInternalMode] = useState<'manual' | 'orbit' | 'transect'>('manual');
  const [internalGimbal, setInternalGimbal] = useState<boolean>(false);

  const activeMode = controlledMode ?? internalMode;
  const isGimbal = controlledGimbal ?? internalGimbal;

  const handleModeSelect = (m: 'manual' | 'orbit' | 'transect') => {
    setInternalMode(m);
    onAutopilotModeChange?.(m);
  };

  const handleGimbalToggle = () => {
    setInternalGimbal((prev) => !prev);
    onCameraGimbalToggle?.();
  };

  // Keyboard input listener with autopilot shortcut dispatch (1/2/3/G)
  const inputRef = useDroneInput(true, {
    onModeSelect: handleModeSelect,
    onGimbalToggle: handleGimbalToggle,
  });

  // Physics state vectors
  const velocity = useRef(new THREE.Vector3(0, 0, 0));
  const yaw = useRef(0);
  const pitch = useRef(0);
  const roll = useRef(0);
  const propRotation = useRef(0);

  // Autopilot trajectory state (§4.3)
  const orbitAngle = useRef<number>(0);
  const transectDir = useRef<1 | -1>(1);

  // Prop spin animation state for DroneModel
  const [propAngle, setPropAngle] = useState(0);

  // Proximity sensor telemetry state for soft-bump response
  const sensorReadingsRef = useRef<SensorReadings>({ left: 50, right: 50, bottom: 50 });

  // World dimensions
  const worldW = Math.max(1, (meshStats?.width ?? 512) * 0.1);
  const worldD = Math.max(1, (meshStats?.height ?? 512) * 0.1);

  // Compute initial safe spawn altitude above terrain surface
  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;
  const maxElevWorld = isRelative
    ? 18.0 * verticalScale
    : (meshStats.elevation_max - meshStats.elevation_min) * verticalScale * 0.1;

  const initialX = spawnPoint ? spawnPoint[0] : 0;
  const initialZ = spawnPoint ? spawnPoint[2] : 0;
  const spawnGroundY = getTerrainElevationAt(
    initialX,
    initialZ,
    dsmRaw,
    meshStats,
    verticalScale,
    waterLevel
  );
  const initialY = (spawnPoint && spawnPoint[1] > 0)
    ? spawnPoint[1]
    : Math.max(16, spawnGroundY + 10.0, maxElevWorld + 8.0);

  // Initialize drone transform on mount
  useLayoutEffect(() => {
    if (!droneGroupRef.current) return;
    droneGroupRef.current.position.set(initialX, initialY, initialZ);
    droneGroupRef.current.rotation.set(0, 0, 0, 'YXZ');
    velocity.current.set(0, 0, 0);
    yaw.current = 0;
    pitch.current = 0;
    roll.current = 0;

    const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
    camera.position.copy(droneGroupRef.current.position).add(noseOffset);
    camera.quaternion.copy(droneGroupRef.current.quaternion);
    camera.updateMatrixWorld();
  }, [initialX, initialY, initialZ, camera]);

  // Main 60 FPS flight physics loop
  useFrame((_, delta) => {
    if (!droneGroupRef.current) return;

    const dt = Math.min(delta, 0.1);
    const input = inputRef.current;

    // Pilot safety override: any manual thrust input immediately disengages autopilot
    const hasManualThrust = input.pitchForward || input.pitchBackward || input.throttleUp || input.throttleDown || input.yawLeft || input.yawRight;
    if (hasManualThrust && activeMode !== 'manual') {
      handleModeSelect('manual');
    }

    const currentYaw = yaw.current;
    const forwardVec = new THREE.Vector3(-Math.sin(currentYaw), 0, -Math.cos(currentYaw));
    const rightVec = new THREE.Vector3(Math.cos(currentYaw), 0, -Math.sin(currentYaw));

    let currentSpeed = 0;
    let groundWorldY = 0;

    if (activeMode === 'orbit') {
      // 1. Orbital Point-of-Interest (POI) Recon Autopilot Mode (§4.3)
      // Circles the terrain center at fixed radius and safe cruise altitude
      const orbitRadius = Math.min(worldW, worldD) * 0.36;
      const orbitSpeed = 0.26; // rad/s
      orbitAngle.current += orbitSpeed * dt;

      const targetX = orbitRadius * Math.cos(orbitAngle.current);
      const targetZ = orbitRadius * Math.sin(orbitAngle.current);
      groundWorldY = getTerrainElevationAt(targetX, targetZ, dsmRaw, meshStats, verticalScale, waterLevel);
      const targetY = Math.max(maxElevWorld + 6.0, groundWorldY + 6.0);

      droneGroupRef.current.position.lerp(new THREE.Vector3(targetX, targetY, targetZ), 5 * dt);

      // Tangent heading with smooth banking into orbit
      const targetYaw = -orbitAngle.current - Math.PI / 2;
      yaw.current = THREE.MathUtils.lerp(yaw.current, targetYaw, 4 * dt);
      pitch.current = THREE.MathUtils.lerp(pitch.current, -0.04, 4 * dt);
      roll.current = THREE.MathUtils.lerp(roll.current, -0.16, 5 * dt);

      currentSpeed = orbitRadius * orbitSpeed;
      velocity.current.set(-orbitRadius * Math.sin(orbitAngle.current) * orbitSpeed, 0, orbitRadius * Math.cos(orbitAngle.current) * orbitSpeed);
    } else if (activeMode === 'transect') {
      // 2. Terrain-Hugging Transect Tour Autopilot Mode (§4.3)
      // Cruises West-to-East across tile centerline, hugging terrain at +5.5m AGL
      const transectSpeed = 8.5; // m/s
      const limitX = worldW * 0.40;

      let nextX = droneGroupRef.current.position.x + transectDir.current * transectSpeed * dt;
      if (nextX > limitX) {
        transectDir.current = -1;
        nextX = limitX;
      } else if (nextX < -limitX) {
        transectDir.current = 1;
        nextX = -limitX;
      }

      droneGroupRef.current.position.x = nextX;
      droneGroupRef.current.position.z = THREE.MathUtils.lerp(droneGroupRef.current.position.z, 0, 2 * dt);

      groundWorldY = getTerrainElevationAt(
        droneGroupRef.current.position.x,
        droneGroupRef.current.position.z,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel
      );
      const targetY = groundWorldY + 5.5;
      droneGroupRef.current.position.y = THREE.MathUtils.lerp(droneGroupRef.current.position.y, targetY, 4 * dt);

      const targetYaw = transectDir.current > 0 ? -Math.PI / 2 : Math.PI / 2;
      yaw.current = THREE.MathUtils.lerp(yaw.current, targetYaw, 4 * dt);
      pitch.current = THREE.MathUtils.lerp(pitch.current, -0.05, 4 * dt);
      roll.current = THREE.MathUtils.lerp(roll.current, 0, 4 * dt);

      currentSpeed = transectSpeed;
      velocity.current.set(transectDir.current * transectSpeed, (targetY - droneGroupRef.current.position.y) * 2, 0);
    } else {
      // 3. Manual Flight Mode (Phases 2 - 4)
      const turboMult = input.turbo ? 2.6 : 1.0;
      const baseThrust = 45.0 * turboMult;
      const thrustVec = new THREE.Vector3(0, 0, 0);

      // Forward / Backward thrust (W / S)
      if (input.pitchForward) {
        thrustVec.add(forwardVec.clone().multiplyScalar(baseThrust));
      }
      if (input.pitchBackward) {
        thrustVec.sub(forwardVec.clone().multiplyScalar(baseThrust * 0.65));
      }

      // Vertical Ascent / Descent throttle (Space / Q / E)
      let vertThrust = 0;
      if (input.throttleUp) vertThrust += 38.0 * turboMult;
      if (input.throttleDown) vertThrust -= 28.0 * turboMult;
      thrustVec.y += vertThrust;

      // Yaw rotation (A / D)
      let yawRate = 0;
      const yawSpeed = input.turbo ? 2.4 : 1.7;
      if (input.yawLeft) yawRate += yawSpeed;
      if (input.yawRight) yawRate -= yawSpeed;
      yaw.current += yawRate * dt;

      if (yaw.current > Math.PI * 2) yaw.current -= Math.PI * 2;
      if (yaw.current < 0) yaw.current += Math.PI * 2;

      // Newtonian velocity integration
      const dragCoeff = 2.8;
      const accel = thrustVec.clone().sub(velocity.current.clone().multiplyScalar(dragCoeff));
      velocity.current.add(accel.multiplyScalar(dt));

      // Soft-bump velocity zeroing from proximity sensors (§5)
      const sensors = sensorReadingsRef.current;
      if (sensors.right < 1.5) {
        const latSpeed = velocity.current.dot(rightVec);
        if (latSpeed > 0) velocity.current.sub(rightVec.clone().multiplyScalar(latSpeed));
      }
      if (sensors.left < 1.5) {
        const latSpeed = velocity.current.dot(rightVec);
        if (latSpeed < 0) velocity.current.sub(rightVec.clone().multiplyScalar(latSpeed));
      }
      if (sensors.bottom < 1.5 && velocity.current.y < 0) {
        velocity.current.y = 0;
      }

      // Terrain Clearance & Collision Resolution (§5)
      const prevPos = droneGroupRef.current.position.clone();
      const candidatePos = prevPos.clone().add(velocity.current.clone().multiplyScalar(dt));

      const collision = resolveTerrainCollision(
        prevPos,
        candidatePos,
        velocity.current,
        dsmRaw,
        meshStats,
        verticalScale,
        waterLevel,
        1.2
      );

      droneGroupRef.current.position.set(
        collision.position.x,
        collision.position.y,
        collision.position.z
      );
      velocity.current.set(
        collision.velocity.x,
        collision.velocity.y,
        collision.velocity.z
      );
      groundWorldY = collision.groundWorldY;

      // Dynamic Pitch and Auto-bank Roll coupling
      const forwardSpeed = velocity.current.dot(forwardVec);
      const lateralSpeed = velocity.current.dot(rightVec);
      const targetPitch = Math.max(-0.26, Math.min(0.18, -forwardSpeed * 0.012));
      pitch.current = THREE.MathUtils.lerp(pitch.current, targetPitch, 8 * dt);

      const targetRoll = Math.max(-0.45, Math.min(0.45, -lateralSpeed * 0.04 - yawRate * 0.12));
      roll.current = THREE.MathUtils.lerp(roll.current, targetRoll, 10 * dt);

      currentSpeed = velocity.current.length();
    }

    // Apply drone body orientation in YXZ Euler order
    droneGroupRef.current.rotation.set(pitch.current, yaw.current, roll.current, 'YXZ');

    // Camera synchronization: Selectable Gimbal vs Rigid Nose-Lock (§7)
    const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
    noseOffset.applyQuaternion(droneGroupRef.current.quaternion);
    camera.position.copy(droneGroupRef.current.position).add(noseOffset);

    if (isGimbal) {
      // 2-axis horizon-stabilized camera: follow heading, keep roll/pitch level
      const gimbalEuler = new THREE.Euler(0, yaw.current, 0, 'YXZ');
      camera.quaternion.setFromEuler(gimbalEuler);
    } else {
      // Rigid nose-locked FPV camera
      camera.quaternion.copy(droneGroupRef.current.quaternion);
    }

    // Propeller spin speed tied to throttle magnitude
    const spinRate = (hasManualThrust || activeMode !== 'manual' ? (input.turbo ? 75 : 50) : 18) + currentSpeed * 0.7;
    propRotation.current += spinRate * dt;
    setPropAngle(propRotation.current);

    // Emit live telemetry to HUD
    if (onTelemetryUpdate) {
      let headingDeg = THREE.MathUtils.radToDeg(-yaw.current);
      if (headingDeg < 0) headingDeg += 360;

      const aglMeters = computeAgl(
        droneGroupRef.current.position.y,
        groundWorldY,
        meshStats,
        verticalScale
      );
      const mslMeters = computeMsl(
        droneGroupRef.current.position.y,
        meshStats,
        verticalScale
      );

      onTelemetryUpdate({
        speed: currentSpeed,
        verticalSpeed: velocity.current.y,
        altitudeMsl: mslMeters,
        altitudeAgl: aglMeters,
        heading: Math.round(headingDeg),
        pitch: THREE.MathUtils.radToDeg(pitch.current),
        roll: THREE.MathUtils.radToDeg(roll.current),
        gridX: droneGroupRef.current.position.x,
        gridZ: droneGroupRef.current.position.z,
        sensors: {
          left: sensorReadingsRef.current.left,
          right: sensorReadingsRef.current.right,
          bottom: aglMeters,
        },
        autopilotMode: activeMode,
        cameraGimbal: isGimbal,
      });
    }
  });

  return (
    <group ref={droneGroupRef}>
      <DroneModel
        throttle={velocity.current.length()}
        propRotation={propAngle}
      />
      <ProximitySensors
        droneRef={droneGroupRef}
        dsmRaw={dsmRaw}
        meshStats={meshStats}
        verticalScale={verticalScale}
        waterLevel={waterLevel}
        onReadingsUpdate={(r) => {
          sensorReadingsRef.current = r;
        }}
      />
    </group>
  );
}
