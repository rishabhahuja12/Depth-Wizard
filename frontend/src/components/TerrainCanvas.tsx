import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface TerrainCanvasProps {
  heightmapB64: string;
  rgbB64: string;
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
    elevation_range: number;
  };
  verticalScale: number;
  waterLevel: number;
  showContours: boolean;
  dsmRaw: number[][];
}

function Terrain({ heightmapB64, rgbB64, meshStats, verticalScale, waterLevel }: TerrainCanvasProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [heightTex, setHeightTex] = useState<THREE.Texture | null>(null);
  const [colorTex, setColorTex] = useState<THREE.Texture | null>(null);

  const loader = useMemo(() => new THREE.TextureLoader(), []);

  useEffect(() => {
    // Load RGB texture
    const rgbUrl = `data:image/jpeg;base64,${rgbB64}`;
    loader.load(rgbUrl, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      setColorTex(tex);
    });

    // Load heightmap texture
    const hmUrl = `data:image/png;base64,${heightmapB64}`;
    loader.load(hmUrl, (tex) => {
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      setHeightTex(tex);
    });
  }, [heightmapB64, rgbB64, loader]);

  const geometry = useMemo(() => {
    const w = meshStats.width;
    const h = meshStats.height;
    const segments = Math.min(w, 512);
    return new THREE.PlaneGeometry(w * 0.1, h * 0.1, segments, segments);
  }, [meshStats]);

  if (!heightTex || !colorTex) return null;

  const displacementScale = meshStats.elevation_range * verticalScale * 0.1;

  return (
    <>
      <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <primitive object={geometry} />
        <meshStandardMaterial
          map={colorTex}
          displacementMap={heightTex}
          displacementScale={displacementScale}
          displacementBias={0}
          side={THREE.DoubleSide}
          roughness={0.8}
          metalness={0.1}
        />
      </mesh>

      {/* Water plane for flood simulation */}
      {waterLevel > 0 && (
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, waterLevel * verticalScale * 0.1, 0]}
        >
          <planeGeometry args={[meshStats.width * 0.12, meshStats.height * 0.12]} />
          <meshStandardMaterial
            color="#0077be"
            transparent
            opacity={0.5}
            side={THREE.DoubleSide}
            roughness={0.1}
            metalness={0.3}
          />
        </mesh>
      )}
    </>
  );
}

function CameraController() {
  const { camera, gl } = useThree();
  const keys = useRef<Set<string>>(new Set());
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const isLocked = useRef(false);

  useEffect(() => {
    camera.position.set(0, 30, 40);
    camera.lookAt(0, 0, 0);

    const onKeyDown = (e: KeyboardEvent) => keys.current.add(e.key.toLowerCase());
    const onKeyUp = (e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase());

    const onMouseMove = (e: MouseEvent) => {
      if (!isLocked.current) return;
      euler.current.y -= e.movementX * 0.002;
      euler.current.x -= e.movementY * 0.002;
      euler.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, euler.current.x));
      camera.quaternion.setFromEuler(euler.current);
    };

    const onClick = () => {
      gl.domElement.requestPointerLock();
    };

    const onPointerLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    document.addEventListener('mousemove', onMouseMove);
    gl.domElement.addEventListener('click', onClick);
    document.addEventListener('pointerlockchange', onPointerLockChange);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      document.removeEventListener('mousemove', onMouseMove);
      gl.domElement.removeEventListener('click', onClick);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
    };
  }, [camera, gl]);

  useFrame((_, delta) => {
    const speed = keys.current.has('shift') ? 60 : 20;
    const dir = new THREE.Vector3();

    if (keys.current.has('w')) dir.z -= 1;
    if (keys.current.has('s')) dir.z += 1;
    if (keys.current.has('a')) dir.x -= 1;
    if (keys.current.has('d')) dir.x += 1;
    if (keys.current.has('q') || keys.current.has(' ')) dir.y += 1;
    if (keys.current.has('e')) dir.y -= 1;

    if (dir.lengthSq() > 0) {
      dir.normalize();
      dir.applyQuaternion(camera.quaternion);
      camera.position.addScaledVector(dir, speed * delta);
    }

    // Clamp minimum height
    if (camera.position.y < 2) camera.position.y = 2;
  });

  return null;
}

export default function TerrainCanvas(props: TerrainCanvasProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ fov: 60, near: 0.1, far: 2000 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#0a0f1a' }}
      >
        <fog attach="fog" args={['#0a0f1a', 50, 200]} />
        <ambientLight intensity={0.4} />
        <directionalLight position={[50, 80, 50]} intensity={1.2} castShadow />
        <directionalLight position={[-30, 40, -30]} intensity={0.3} />

        <Terrain {...props} />
        <CameraController />

        <gridHelper args={[200, 50, '#1e293b', '#1e293b']} position={[0, -0.1, 0]} />
      </Canvas>

      {/* HUD overlay */}
      <div className="absolute bottom-4 left-4 glass-panel px-4 py-2 text-xs text-slate-400 space-y-1">
        <p><span className="text-blue-400 font-medium">Click</span> canvas to lock mouse</p>
        <p><span className="text-blue-400 font-medium">WASD</span> move · <span className="text-blue-400 font-medium">Q/E</span> up/down · <span className="text-blue-400 font-medium">Shift</span> sprint</p>
        <p><span className="text-blue-400 font-medium">ESC</span> release mouse</p>
      </div>
    </div>
  );
}
