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
 * DroneFlightController (Phase 2):
 * Full Newtonian flight physics with velocity integration, linear drag,
 * aerodynamic auto-banking into turns, thrust-pitch coupling, and dynamic prop spin.
 */
export default function DroneFlightController({
  spawnPoint,
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel,
  onTelemetryUpdate,
}: DroneFlightControllerProps) {
  const droneGroupRef = useRef<THREE.Group>(null);
  const { camera } = useThree();

  // Keyboard input listener
  const inputRef = useDroneInput(true);

  // Physics state vectors
  const velocity = useRef(new THREE.Vector3(0, 0, 0));
  const yaw = useRef(0);
  const pitch = useRef(0);
  const roll = useRef(0);
  const propRotation = useRef(0);

  // Prop spin animation state for DroneModel
  const [propAngle, setPropAngle] = useState(0);

  // Proximity sensor telemetry state for soft-bump response
  const sensorReadingsRef = useRef<SensorReadings>({ left: 50, right: 50, bottom: 50 });

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
  const initialY = spawnPoint
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

    // 1. Directional vectors in current heading (Yaw decoupled)
    const currentYaw = yaw.current;
    const forwardVec = new THREE.Vector3(-Math.sin(currentYaw), 0, -Math.cos(currentYaw));
    const rightVec = new THREE.Vector3(Math.cos(currentYaw), 0, -Math.sin(currentYaw));

    // 2. Thrust integration with Turbo boost (§4)
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

    // 3. Yaw rotation with smooth angular rate (A / D)
    let yawRate = 0;
    const yawSpeed = input.turbo ? 2.4 : 1.7;
    if (input.yawLeft) yawRate += yawSpeed;
    if (input.yawRight) yawRate -= yawSpeed;
    yaw.current += yawRate * dt;

    // Keep yaw normalized in [0, 2*PI)
    if (yaw.current > Math.PI * 2) yaw.current -= Math.PI * 2;
    if (yaw.current < 0) yaw.current += Math.PI * 2;

    // 4. Newtonian velocity integration with linear aerodynamic drag (§4)
    // a = (F / m) - (k_drag * v)
    const dragCoeff = 2.8;
    const accel = thrustVec.clone().sub(velocity.current.clone().multiplyScalar(dragCoeff));
    velocity.current.add(accel.multiplyScalar(dt));

    // 5. Soft-bump velocity zeroing from proximity sensors (§5)
    // If obstacle is closer than 1.5m, cancel velocity component towards obstacle
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

    // 6. Terrain Clearance & Collision Resolution (§5)
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
      1.2 // Minimum hard-floor & rooftop clearance
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

    // 7. Thrust-pitch coupling (§4): nose dips 5-15 deg under forward acceleration
    const forwardSpeed = velocity.current.dot(forwardVec);
    const lateralSpeed = velocity.current.dot(rightVec);
    const targetPitch = Math.max(-0.26, Math.min(0.18, -forwardSpeed * 0.012));
    pitch.current = THREE.MathUtils.lerp(pitch.current, targetPitch, 8 * dt);

    // 8. Auto-bank roll into turns (§4): roll proportional to lateral velocity and yaw rate
    const targetRoll = Math.max(-0.45, Math.min(0.45, -lateralSpeed * 0.04 - yawRate * 0.12));
    roll.current = THREE.MathUtils.lerp(roll.current, targetRoll, 10 * dt);

    // Apply orientation in YXZ Euler order
    droneGroupRef.current.rotation.set(pitch.current, yaw.current, roll.current, 'YXZ');

    // 9. Rigid nose-locked FPV camera: rigidly pinned to front nose pod
    const noseOffset = new THREE.Vector3(0, 0.08, -0.38);
    noseOffset.applyQuaternion(droneGroupRef.current.quaternion);
    camera.position.copy(droneGroupRef.current.position).add(noseOffset);
    camera.quaternion.copy(droneGroupRef.current.quaternion);

    // 10. Propeller spin speed tied to throttle magnitude (§4)
    const currentSpeed = velocity.current.length();
    const hasThrustInput = input.pitchForward || input.pitchBackward || input.throttleUp || input.throttleDown || Math.abs(yawRate) > 0;
    const spinRate = (hasThrustInput ? (input.turbo ? 75 : 45) : 18) + currentSpeed * 0.7;
    propRotation.current += spinRate * dt;
    setPropAngle(propRotation.current);

    // 11. Emit telemetry to HUD
    if (onTelemetryUpdate) {
      // Heading in degrees: 0 deg North (-Z), 90 deg East (+X), 180 deg South (+Z), 270 deg West (-X)
      let headingDeg = THREE.MathUtils.radToDeg(-yaw.current);
      if (headingDeg < 0) headingDeg += 360;

      const aglMeters = computeAgl(
        collision.position.y,
        collision.groundWorldY,
        meshStats,
        verticalScale
      );
      const mslMeters = computeMsl(
        collision.position.y,
        meshStats,
        verticalScale
      );

      onTelemetryUpdate({
        speed: currentSpeed,
        altitudeMsl: mslMeters,
        altitudeAgl: aglMeters,
        heading: Math.round(headingDeg),
        pitch: THREE.MathUtils.radToDeg(pitch.current),
        roll: THREE.MathUtils.radToDeg(roll.current),
        sensors: {
          left: sensors.left,
          right: sensors.right,
          bottom: aglMeters,
        },
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
