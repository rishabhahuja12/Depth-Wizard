import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { castSensor, MeshElevationStats } from './dsmSampling.ts';

export interface SensorReadings {
  left: number;
  right: number;
  bottom: number;
}

export interface ProximitySensorsProps {
  droneRef: React.RefObject<THREE.Group | null>;
  dsmRaw: number[][];
  meshStats: MeshElevationStats;
  verticalScale?: number;
  waterLevel?: number;
  showVisuals?: boolean;
  onReadingsUpdate?: (readings: SensorReadings) => void;
}

function getSensorColor(distance: number): THREE.Color {
  if (distance < 3.0) return new THREE.Color('#ef4444'); // Red: Alert < 3m
  if (distance <= 10.0) return new THREE.Color('#fbbf24'); // Yellow: Warning 3-10m
  return new THREE.Color('#10b981'); // Green: Clear > 10m
}

/**
 * 3-Sensor Proximity Array (§5):
 * Casts client-side ray-marches Left (-X), Right (+X), and Bottom (-Y) into dsm_raw.
 * Renders in-scene color-coded laser lines with terrain impact markers and reports live telemetry.
 */
export default function ProximitySensors({
  droneRef,
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel,
  showVisuals = false,
  onReadingsUpdate,
}: ProximitySensorsProps) {
  const { camera } = useThree();

  // Laser line geometries with dynamic 2-vertex buffer attributes
  const leftGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);

  const rightGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);

  const bottomGeo = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    return geo;
  }, []);

  // Three.js Line objects
  const leftLine = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#10b981', transparent: true, opacity: 0.85 });
    return new THREE.Line(leftGeo, mat);
  }, [leftGeo]);

  const rightLine = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#10b981', transparent: true, opacity: 0.85 });
    return new THREE.Line(rightGeo, mat);
  }, [rightGeo]);

  const bottomLine = useMemo(() => {
    const mat = new THREE.LineBasicMaterial({ color: '#10b981', transparent: true, opacity: 0.85 });
    return new THREE.Line(bottomGeo, mat);
  }, [bottomGeo]);

  useEffect(() => {
    return () => {
      leftGeo.dispose();
      rightGeo.dispose();
      bottomGeo.dispose();
      (leftLine.material as THREE.Material).dispose();
      (rightLine.material as THREE.Material).dispose();
      (bottomLine.material as THREE.Material).dispose();
    };
  }, [leftGeo, rightGeo, bottomGeo, leftLine, rightLine, bottomLine]);

  // Impact dot mesh refs
  const leftDotRef = useRef<THREE.Mesh>(null);
  const rightDotRef = useRef<THREE.Mesh>(null);
  const bottomDotRef = useRef<THREE.Mesh>(null);

  const leftDotMatRef = useRef<THREE.MeshBasicMaterial>(null);
  const rightDotMatRef = useRef<THREE.MeshBasicMaterial>(null);
  const bottomDotMatRef = useRef<THREE.MeshBasicMaterial>(null);

  // Guard against near-plane WebGL projection inversion
  const isBehindCameraNearPlane = (pt: THREE.Vector3) => {
    const pCam = pt.clone().applyMatrix4(camera.matrixWorldInverse);
    return pCam.z >= -(camera.near + 0.05);
  };

  useFrame(() => {
    if (!droneRef.current) return;

    const drone = droneRef.current;
    drone.updateMatrixWorld();

    // 1. Left Sensor (Port arm anchor)
    const leftAnchorLocal = new THREE.Vector3(-0.65, 0.02, 0);
    const leftOrigin = leftAnchorLocal.applyMatrix4(drone.matrixWorld);
    const leftDir = new THREE.Vector3(-1, 0, 0)
      .applyQuaternion(drone.quaternion)
      .normalize();
    const leftDist = castSensor(
      leftOrigin,
      leftDir,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      50,
      0.5
    );
    const leftHit = leftOrigin.clone().add(leftDir.clone().multiplyScalar(leftDist));

    // Update Left Line Buffer
    if (showVisuals) {
      const leftPos = leftGeo.attributes.position as THREE.BufferAttribute;
      const leftArr = leftPos.array as Float32Array;
      leftArr[0] = leftOrigin.x;
      leftArr[1] = leftOrigin.y;
      leftArr[2] = leftOrigin.z;
      leftArr[3] = leftHit.x;
      leftArr[4] = leftHit.y;
      leftArr[5] = leftHit.z;
      leftPos.needsUpdate = true;

      const leftColor = getSensorColor(leftDist);
      (leftLine.material as THREE.LineBasicMaterial).color.copy(leftColor);
      leftLine.visible = !isBehindCameraNearPlane(leftOrigin) && !isBehindCameraNearPlane(leftHit);
      if (leftDotRef.current && leftDotMatRef.current) {
        leftDotRef.current.position.copy(leftHit);
        leftDotRef.current.visible = leftDist < 50 && !isBehindCameraNearPlane(leftHit);
        leftDotMatRef.current.color.copy(leftColor);
      }
    } else {
      leftLine.visible = false;
      if (leftDotRef.current) leftDotRef.current.visible = false;
    }

    // 2. Right Sensor (Starboard arm anchor)
    const rightAnchorLocal = new THREE.Vector3(0.65, 0.02, 0);
    const rightOrigin = rightAnchorLocal.applyMatrix4(drone.matrixWorld);
    const rightDir = new THREE.Vector3(1, 0, 0)
      .applyQuaternion(drone.quaternion)
      .normalize();
    const rightDist = castSensor(
      rightOrigin,
      rightDir,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      50,
      0.5
    );
    const rightHit = rightOrigin.clone().add(rightDir.clone().multiplyScalar(rightDist));

    // Update Right Line Buffer
    if (showVisuals) {
      const rightPos = rightGeo.attributes.position as THREE.BufferAttribute;
      const rightArr = rightPos.array as Float32Array;
      rightArr[0] = rightOrigin.x;
      rightArr[1] = rightOrigin.y;
      rightArr[2] = rightOrigin.z;
      rightArr[3] = rightHit.x;
      rightArr[4] = rightHit.y;
      rightArr[5] = rightHit.z;
      rightPos.needsUpdate = true;

      const rightColor = getSensorColor(rightDist);
      (rightLine.material as THREE.LineBasicMaterial).color.copy(rightColor);
      rightLine.visible = !isBehindCameraNearPlane(rightOrigin) && !isBehindCameraNearPlane(rightHit);
      if (rightDotRef.current && rightDotMatRef.current) {
        rightDotRef.current.position.copy(rightHit);
        rightDotRef.current.visible = rightDist < 50 && !isBehindCameraNearPlane(rightHit);
        rightDotMatRef.current.color.copy(rightColor);
      }
    } else {
      rightLine.visible = false;
      if (rightDotRef.current) rightDotRef.current.visible = false;
    }

    // 3. Bottom Sensor (Belly mount, straight down to ground)
    const bottomAnchorLocal = new THREE.Vector3(0, -0.06, 0);
    const bottomOrigin = bottomAnchorLocal.applyMatrix4(drone.matrixWorld);
    const bottomDir = new THREE.Vector3(0, -1, 0);
    const bottomDist = castSensor(
      bottomOrigin,
      bottomDir,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel,
      50,
      0.5
    );
    const bottomHit = bottomOrigin.clone().add(bottomDir.clone().multiplyScalar(bottomDist));

    // Update Bottom Line Buffer
    if (showVisuals) {
      const bottomPos = bottomGeo.attributes.position as THREE.BufferAttribute;
      const bottomArr = bottomPos.array as Float32Array;
      bottomArr[0] = bottomOrigin.x;
      bottomArr[1] = bottomOrigin.y;
      bottomArr[2] = bottomOrigin.z;
      bottomArr[3] = bottomHit.x;
      bottomArr[4] = bottomHit.y;
      bottomArr[5] = bottomHit.z;
      bottomPos.needsUpdate = true;

      const bottomColor = getSensorColor(bottomDist);
      (bottomLine.material as THREE.LineBasicMaterial).color.copy(bottomColor);
      bottomLine.visible = !isBehindCameraNearPlane(bottomOrigin) && !isBehindCameraNearPlane(bottomHit);
      if (bottomDotRef.current && bottomDotMatRef.current) {
        bottomDotRef.current.position.copy(bottomHit);
        bottomDotRef.current.visible = bottomDist < 50 && !isBehindCameraNearPlane(bottomHit);
        bottomDotMatRef.current.color.copy(bottomColor);
      }
    } else {
      bottomLine.visible = false;
      if (bottomDotRef.current) bottomDotRef.current.visible = false;
    }

    // Report readings to callback
    if (onReadingsUpdate) {
      onReadingsUpdate({
        left: leftDist,
        right: rightDist,
        bottom: bottomDist,
      });
    }
  });

  return (
    <group visible={showVisuals}>
      {/* Left Sensor Laser Line */}
      <primitive object={leftLine} />
      <mesh ref={leftDotRef} visible={false}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial ref={leftDotMatRef} color="#10b981" />
      </mesh>

      {/* Right Sensor Laser Line */}
      <primitive object={rightLine} />
      <mesh ref={rightDotRef} visible={false}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial ref={rightDotMatRef} color="#10b981" />
      </mesh>

      {/* Bottom Sensor Laser Line */}
      <primitive object={bottomLine} />
      <mesh ref={bottomDotRef} visible={false}>
        <sphereGeometry args={[0.08, 8, 8]} />
        <meshBasicMaterial ref={bottomDotMatRef} color="#10b981" />
      </mesh>
    </group>
  );
}
