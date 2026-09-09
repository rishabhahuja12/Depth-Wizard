import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

interface TerrainCanvasProps {
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
  showContours: boolean;
  contourInterval?: number;
  dsmRaw: number[][];
}

function Terrain({
  heightmapB64,
  rgbB64,
  normalMapB64,
  meshStats,
  verticalScale,
  waterLevel,
  showContours,
  contourInterval = 5,
}: TerrainCanvasProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [heightTex, setHeightTex] = useState<THREE.Texture | null>(null);
  const [colorTex, setColorTex] = useState<THREE.Texture | null>(null);
  const [normalTex, setNormalTex] = useState<THREE.Texture | null>(null);

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

    // Load normal map for WebGL dynamic lighting & relief shading
    if (normalMapB64) {
      const normUrl = `data:image/png;base64,${normalMapB64}`;
      loader.load(normUrl, (tex) => {
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        setNormalTex(tex);
      });
    } else {
      setNormalTex(null);
    }
  }, [heightmapB64, rgbB64, normalMapB64, loader]);

  // Clean up GPU texture memory to prevent leaks
  useEffect(() => {
    return () => {
      if (colorTex) colorTex.dispose();
      if (heightTex) heightTex.dispose();
      if (normalTex) normalTex.dispose();
    };
  }, [colorTex, heightTex, normalTex]);

  const geometry = useMemo(() => {
    const w = meshStats.width;
    const h = meshStats.height;
    const segments = Math.min(w, 512);
    return new THREE.PlaneGeometry(w * 0.1, h * 0.1, segments, segments);
  }, [meshStats]);

  useEffect(() => {
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  // WebGL shader uniforms for dynamic 3D contour lines
  const uniformsRef = useRef({
    uShowContours: { value: showContours ? 1.0 : 0.0 },
    uContourInterval: { value: contourInterval },
    uElevationMin: { value: meshStats.elevation_min },
    uVerticalScale: { value: verticalScale },
  });

  useEffect(() => {
    uniformsRef.current.uShowContours.value = showContours ? 1.0 : 0.0;
    uniformsRef.current.uContourInterval.value = contourInterval;
    uniformsRef.current.uElevationMin.value = meshStats.elevation_min;
    uniformsRef.current.uVerticalScale.value = verticalScale;
  }, [showContours, contourInterval, meshStats.elevation_min, verticalScale]);

  const onBeforeCompile = useMemo(() => {
    return (shader: THREE.WebGLProgramParametersWithUniforms) => {
      shader.uniforms.uShowContours = uniformsRef.current.uShowContours;
      shader.uniforms.uContourInterval = uniformsRef.current.uContourInterval;
      shader.uniforms.uElevationMin = uniformsRef.current.uElevationMin;
      shader.uniforms.uVerticalScale = uniformsRef.current.uVerticalScale;

      shader.vertexShader = `
        varying vec3 vTerrainWorldPos;
        ${shader.vertexShader}
      `.replace(
        '#include <worldpos_vertex>',
        `
        #include <worldpos_vertex>
        vTerrainWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        `
      );

      shader.fragmentShader = `
        varying vec3 vTerrainWorldPos;
        uniform float uShowContours;
        uniform float uContourInterval;
        uniform float uElevationMin;
        uniform float uVerticalScale;
        ${shader.fragmentShader}
      `.replace(
        '#include <dithering_fragment>',
        `
        #include <dithering_fragment>
        if (uShowContours > 0.5 && uContourInterval > 0.01) {
          float vScale = max(uVerticalScale * 0.1, 0.0001);
          float elev = uElevationMin + (vTerrainWorldPos.y / vScale);
          float cInt = max(uContourInterval, 0.01);
          float numIntervals = elev / cInt;
          float dist = abs(numIntervals - floor(numIntervals + 0.5)) * cInt;
          float dElev = max(fwidth(elev), 0.01);
          float line = 1.0 - smoothstep(0.0, dElev * 1.5, dist);

          float indexInt = cInt * 5.0;
          float numIndex = elev / indexInt;
          float indexDist = abs(numIndex - floor(numIndex + 0.5)) * indexInt;
          float indexLine = 1.0 - smoothstep(0.0, dElev * 2.2, indexDist);

          vec3 contourColor = mix(vec3(0.05, 0.1, 0.2), vec3(1.0, 1.0, 1.0), indexLine * 0.5);
          float alpha = max(line * 0.75, indexLine * 0.95);
          gl_FragColor.rgb = mix(gl_FragColor.rgb, contourColor, alpha);
        }
        `
      );
    };
  }, []);

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
          normalMap={normalTex || undefined}
          normalScale={new THREE.Vector2(1.2, 1.2)}
          side={THREE.DoubleSide}
          roughness={0.7}
          metalness={0.1}
          onBeforeCompile={onBeforeCompile}
          customProgramCacheKey={() => 'terrain_contour_mat'}
        />
      </mesh>

      {/* Synchronized Water plane for flood simulation */}
      {waterLevel > meshStats.elevation_min && (
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, (waterLevel - meshStats.elevation_min) * verticalScale * 0.1, 0]}
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
    camera.updateMatrixWorld();
    // Fix first-click camera angle snap: extract current Euler orientation from quaternion
    euler.current.setFromQuaternion(camera.quaternion, 'YXZ');
    euler.current.z = 0;

    const onKeyDown = (e: KeyboardEvent) => keys.current.add(e.key.toLowerCase());
    const onKeyUp = (e: KeyboardEvent) => keys.current.delete(e.key.toLowerCase());

    const onMouseMove = (e: MouseEvent) => {
      if (!isLocked.current) return;
      euler.current.y -= e.movementX * 0.002;
      euler.current.x -= e.movementY * 0.002;
      euler.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, euler.current.x));
      euler.current.z = 0;
      camera.quaternion.setFromEuler(euler.current);
    };

    const onClick = () => {
      gl.domElement.requestPointerLock();
    };

    const onPointerLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
      if (isLocked.current) {
        // Re-sync Euler orientation when pointer lock engages to prevent any angle snap
        euler.current.setFromQuaternion(camera.quaternion, 'YXZ');
        euler.current.z = 0;
      }
    };

    const onBlur = () => {
      keys.current.clear();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    document.addEventListener('mousemove', onMouseMove);
    gl.domElement.addEventListener('click', onClick);
    document.addEventListener('pointerlockchange', onPointerLockChange);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      document.removeEventListener('mousemove', onMouseMove);
      gl.domElement.removeEventListener('click', onClick);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      window.removeEventListener('blur', onBlur);
    };
  }, [camera, gl]);

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1);
    const speed = keys.current.has('shift') ? 60 : 20;

    // Separate horizontal flight from vertical ascension
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

    // World vertical ascension along world Y (Q / Space = Up, E = Down)
    let upDown = 0;
    if (keys.current.has('q') || keys.current.has(' ')) upDown += 1;
    if (keys.current.has('e')) upDown -= 1;

    if (upDown !== 0) {
      camera.position.y += upDown * speed * dt;
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
