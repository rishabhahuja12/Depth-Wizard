import React, { useRef, useLayoutEffect, useMemo, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { voxelize, VoxelGrid } from '../lib/voxelize.ts';

export interface VoxelTerrainProps {
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
  targetResolution?: number;
  bandCount?: number;
  gapRatio?: number;
}

/**
 * Inner VoxelMesh component rendering the InstancedMesh and synchronized water plane.
 * Can be mounted inside any existing React Three Fiber <Canvas>.
 */
export function VoxelMesh({
  dsmRaw,
  meshStats,
  verticalScale = 1.0,
  waterLevel = 0,
  targetResolution = 64,
  bandCount = 8,
  gapRatio = 0.90,
}: VoxelTerrainProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const isRelative = (meshStats?.elevation_range ?? 0) <= 2.0;

  // In metric mode: 1 meter elevation corresponds to 0.1 Three.js world units.
  // In relative mode: normalized elevation [0, 1] scaled to 18.0 world units to match TerrainCanvas.
  const verticalFactor = isRelative ? 18.0 * verticalScale : verticalScale * 0.1;

  // 1. Compute downsampled voxel grid via pure pooling and quantization
  const voxelGrid: VoxelGrid = useMemo(() => {
    return voxelize(dsmRaw, {
      targetResolution,
      bandCount,
      gapRatio,
      elevationMin: meshStats.elevation_min,
      elevationMax: meshStats.elevation_max,
      worldWidth: meshStats.width * 0.1,
      worldDepth: meshStats.height * 0.1,
    });
  }, [
    dsmRaw,
    targetResolution,
    bandCount,
    gapRatio,
    meshStats.elevation_min,
    meshStats.elevation_max,
    meshStats.width,
    meshStats.height,
  ]);

  const blockCount = voxelGrid.blocks.length;

  // 2. Populate InstancedMesh transform matrices and per-instance colors
  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh || blockCount === 0) return;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < blockCount; i++) {
      const block = voxelGrid.blocks[i];
      const normH = block.normalizedHeight;
      const rawH = block.rawHeight;

      // Compute physical block height in Three.js world units
      const hWorld = isRelative
        ? Math.max(0.12, normH * 18.0 * verticalScale)
        : Math.max(0.12, (rawH - meshStats.elevation_min) * verticalScale * 0.1);

      // Vertical anchoring (§3.4): instance position.y is height / 2 so the base rests at y = 0
      dummy.position.set(block.posX, hWorld / 2, block.posZ);
      dummy.scale.set(voxelGrid.cellWidth, hWorld, voxelGrid.cellDepth);
      dummy.updateMatrix();

      mesh.setMatrixAt(i, dummy.matrix);

      // Apply discrete Turbo band color
      color.set(block.colorHex);
      mesh.setColorAt(i, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  }, [
    voxelGrid,
    blockCount,
    isRelative,
    verticalScale,
    meshStats.elevation_min,
  ]);

  if (blockCount === 0) return null;

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[undefined, undefined, blockCount]}
        castShadow
        receiveShadow
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          roughness={0.45}
          metalness={0.08}
          side={THREE.FrontSide}
        />
      </instancedMesh>

      {/* Synchronized Water plane for flood simulation */}
      {waterLevel > meshStats.elevation_min && (
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[
            0,
            (waterLevel - meshStats.elevation_min) * verticalFactor + 0.05,
            0,
          ]}
        >
          <planeGeometry args={[meshStats.width * 0.12, meshStats.height * 0.12]} />
          <meshStandardMaterial
            color="#0077be"
            transparent
            opacity={0.55}
            depthWrite={false}
            side={THREE.DoubleSide}
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      )}
    </>
  );
}

function VoxelCameraController() {
  const { camera, gl } = useThree();
  const keys = useRef<Set<string>>(new Set());
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const isLocked = useRef(false);
  const isDragging = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });

  useEffect(() => {
    camera.position.set(0, 30, 40);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    euler.current.setFromQuaternion(camera.quaternion, 'YXZ');
    euler.current.z = 0;

    const onKeyDown = (e: KeyboardEvent) => keys.current.add(e.key.toLowerCase());
    const onKeyUp = (e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase());

    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 0) {
        isDragging.current = true;
        previousMousePosition.current = { x: e.clientX, y: e.clientY };
      }
    };

    const onMouseUp = () => {
      isDragging.current = false;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (isLocked.current) {
        euler.current.y -= e.movementX * 0.002;
        euler.current.x -= e.movementY * 0.002;
        euler.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, euler.current.x));
        euler.current.z = 0;
        camera.quaternion.setFromEuler(euler.current);
      } else if (isDragging.current) {
        const deltaX = e.clientX - previousMousePosition.current.x;
        const deltaY = e.clientY - previousMousePosition.current.y;
        previousMousePosition.current = { x: e.clientX, y: e.clientY };

        euler.current.y -= deltaX * 0.003;
        euler.current.x -= deltaY * 0.003;
        euler.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, euler.current.x));
        euler.current.z = 0;
        camera.quaternion.setFromEuler(euler.current);
      }
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const dir = new THREE.Vector3();
      camera.getWorldDirection(dir);
      const zoomStep = Math.max(1.0, Math.min(10.0, camera.position.length() * 0.08));
      camera.position.addScaledVector(dir, (e.deltaY > 0 ? -1 : 1) * zoomStep);
      if (camera.position.y < 3.0) camera.position.y = 3.0;
    };

    const onDblClick = () => {
      gl.domElement.requestPointerLock();
    };

    const onPointerLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
      if (isLocked.current) {
        euler.current.setFromQuaternion(camera.quaternion, 'YXZ');
        euler.current.z = 0;
      }
    };

    const onBlur = () => {
      keys.current.clear();
      isDragging.current = false;
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('mouseup', onMouseUp);
    gl.domElement.addEventListener('mousedown', onMouseDown);
    gl.domElement.addEventListener('mousemove', onMouseMove);
    gl.domElement.addEventListener('wheel', onWheel, { passive: false });
    gl.domElement.addEventListener('dblclick', onDblClick);
    document.addEventListener('pointerlockchange', onPointerLockChange);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('mouseup', onMouseUp);
      gl.domElement.removeEventListener('mousedown', onMouseDown);
      gl.domElement.removeEventListener('mousemove', onMouseMove);
      gl.domElement.removeEventListener('wheel', onWheel);
      gl.domElement.removeEventListener('dblclick', onDblClick);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      window.removeEventListener('blur', onBlur);
    };
  }, [camera, gl]);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1);
    const speed = keys.current.has('shift') ? 60 : 20;

    const yaw = euler.current.y;
    const forward = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
    const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    const move = new THREE.Vector3();

    if (keys.current.has('w') || keys.current.has('arrowup')) move.add(forward);
    if (keys.current.has('s') || keys.current.has('arrowdown')) move.sub(forward);
    if (keys.current.has('d') || keys.current.has('arrowright')) move.add(right);
    if (keys.current.has('a') || keys.current.has('arrowleft')) move.sub(right);

    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(speed * dt);
      camera.position.add(move);
    }

    let upDown = 0;
    if (keys.current.has('q') || keys.current.has(' ')) upDown += 1;
    if (keys.current.has('e')) upDown -= 1;

    if (upDown !== 0) {
      camera.position.y += upDown * speed * dt;
    }

    if (camera.position.y < 3.0) camera.position.y = 3.0;
  });

  return null;
}

/**
 * Full VoxelTerrain canvas with lighting, camera flight controls, and telemetry HUD.
 */
export default function VoxelTerrain(props: VoxelTerrainProps) {
  return (
    <div className="w-full h-full relative select-none">
      <Canvas
        camera={{ fov: 60, near: 0.1, far: 2000 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#050608' }}
      >
        <fog attach="fog" args={['#050608', 60, 250]} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[50, 80, 50]} intensity={1.3} castShadow />
        <directionalLight position={[-30, 40, -30]} intensity={0.35} />

        <VoxelMesh {...props} />
        <VoxelCameraController />

        <gridHelper args={[200, 50, '#1e293b', '#1e293b']} position={[0, -0.05, 0]} />
      </Canvas>

      {/* Flight Telemetry HUD */}
      <div className="absolute bottom-4 left-4 p-3 pointer-events-none bg-black/90 backdrop-blur-md border border-white/15 text-xs font-mono space-y-2">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 bg-[#10B981]" />
          <span className="text-[10px] font-mono font-bold text-white tracking-widest uppercase">
            Voxel 3D Navigation
          </span>
        </div>
        <div className="text-[10px] text-neutral-300 space-y-1.5 font-mono">
          <div className="flex items-center gap-2">
            <kbd className="px-1.5 py-0.5 border border-white/20 bg-white/5 text-white font-bold text-[9px]">DRAG</kbd>
            <span className="text-neutral-400">ROTATE</span>
            <kbd className="px-1.5 py-0.5 border border-white/20 bg-white/5 text-white font-bold text-[9px] ml-1">SCROLL</kbd>
            <span className="text-neutral-400">ZOOM</span>
          </div>
          <div className="flex items-center gap-2">
            <kbd className="px-1.5 py-0.5 border border-white/20 bg-white/5 text-white font-bold text-[9px]">WASD</kbd>
            <span className="text-neutral-400">FLY</span>
            <kbd className="px-1.5 py-0.5 border border-white/20 bg-white/5 text-white font-bold text-[9px] ml-1">Q / E</kbd>
            <span className="text-neutral-400">ALTITUDE</span>
          </div>
        </div>
      </div>
    </div>
  );
}
