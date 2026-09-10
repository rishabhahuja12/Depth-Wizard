# DepthWizard — Frontend Implementation Bible (Part 2)

> **Continuation of:** `depthwizard_implementation_bible.md`
> **This file contains:** Complete source code for every frontend file (Tasks 16–26)
> **RULE:** Copy each file exactly as shown. Do NOT modify imports, types, or structure.

---

## FILE: `frontend/src/main.tsx`

```tsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './styles/globals.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

---

## FILE: `frontend/src/styles/globals.css`

```css
@import "tailwindcss";

:root {
  --accent: 217 91% 60%;
  --bg-primary: 222 47% 6%;
  --bg-card: 215 28% 10%;
  --border-subtle: 215 20% 18%;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: hsl(var(--bg-primary));
  color: #e2e8f0;
  overflow: hidden;
  height: 100vh;
}

#root {
  height: 100vh;
}

/* Glassmorphism panel base */
.glass-panel {
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
}

.glass-panel-strong {
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(30px);
  -webkit-backdrop-filter: blur(30px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 16px;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 6px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.15);
  border-radius: 3px;
}

/* Slider styling */
input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  outline: none;
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: hsl(var(--accent));
  cursor: pointer;
  border: 2px solid rgba(255, 255, 255, 0.3);
}
```

---

## FILE: `frontend/src/utils/colormap.ts`

```typescript
/**
 * Turbo colormap: maps a value in [0, 1] to an RGB tuple [r, g, b] each in [0, 255].
 */
export function turboColormap(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  let r: number, g: number, b: number;

  if (t < 0.25) {
    r = 0;
    g = t * 4;
    b = 1;
  } else if (t < 0.5) {
    r = 0;
    g = 1;
    b = 1 - (t - 0.25) * 4;
  } else if (t < 0.75) {
    r = (t - 0.5) * 4;
    g = 1;
    b = 0;
  } else {
    r = 1;
    g = 1 - (t - 0.75) * 4;
    b = 0;
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Generate a full colormap array of N entries.
 */
export function generateColormapLUT(n: number = 256): Uint8Array {
  const lut = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = turboColormap(i / (n - 1));
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}
```

---

## FILE: `frontend/src/hooks/useInference.ts`

```typescript
import { useState, useCallback } from 'react';

export interface MeshStats {
  width: number;
  height: number;
  original_width: number;
  original_height: number;
  vertices: number;
  triangles: number;
  elevation_min: number;
  elevation_max: number;
  elevation_range: number;
  pixel_size: number;
}

export interface CalibrationData {
  alpha: number;
  min: number;
  max: number;
  mean: number;
  mode: string;
  unit: string;
}

export interface InferenceResult {
  request_id: string;
  heightmap_b64: string;
  rgb_b64: string;
  dsm_colorized_b64: string;
  mesh_stats: MeshStats;
  calibration: CalibrationData;
  dsm_raw: number[][];
  is_georef: boolean;
  inference_time_ms: number;
}

export function useInference() {
  const [data, setData] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const upload = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setProgress(10);

    try {
      const formData = new FormData();
      formData.append('file', file);

      setProgress(30);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      setProgress(80);

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || `Server error: ${response.status}`);
      }

      const result: InferenceResult = await response.json();
      setData(result);
      setProgress(100);
    } catch (e: any) {
      setError(e.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setProgress(0);
  }, []);

  return { upload, data, loading, error, progress, reset };
}
```

---

## FILE: `frontend/src/components/Dropzone.tsx`

```tsx
import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, AlertCircle } from 'lucide-react';

interface DropzoneProps {
  onFileSelected: (file: File) => void;
  loading: boolean;
  progress: number;
}

const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/tiff'];
const ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];

export default function Dropzone({ onFileSelected, loading, progress }: DropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported format. Accepted: PNG, JPG, TIFF`);
      return false;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('File too large. Maximum 100 MB.');
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = ALLOWED_EXTENSIONS.join(',');
    input.onchange = (e: any) => {
      const file = e.target.files[0];
      if (file && validateFile(file)) {
        onFileSelected(file);
      }
    };
    input.click();
  };

  return (
    <div className="flex items-center justify-center h-full p-8">
      <div
        className={`
          glass-panel w-full max-w-2xl p-12 text-center cursor-pointer
          transition-all duration-300 ease-out
          ${isDragOver
            ? 'border-blue-500 bg-blue-500/10 scale-[1.02] shadow-[0_0_40px_rgba(59,130,246,0.15)]'
            : 'border-dashed border-2 border-white/10 hover:border-white/20 hover:bg-white/[0.03]'
          }
        `}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={loading ? undefined : handleClick}
      >
        {loading ? (
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
            <p className="text-lg text-blue-400 font-medium">Processing image...</p>
            <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-slate-400">{progress}% — Estimating depth & building terrain</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="w-20 h-20 mx-auto rounded-2xl bg-blue-500/10 flex items-center justify-center">
              {isDragOver ? (
                <ImageIcon className="w-10 h-10 text-blue-400" />
              ) : (
                <Upload className="w-10 h-10 text-blue-400" />
              )}
            </div>
            <div>
              <p className="text-xl font-semibold text-white">
                {isDragOver ? 'Drop image here' : 'Upload satellite imagery'}
              </p>
              <p className="text-sm text-slate-400 mt-2">
                Drag & drop or click to browse — PNG, JPG, or GeoTIFF
              </p>
            </div>
            {error && (
              <div className="flex items-center justify-center gap-2 text-red-400 text-sm">
                <AlertCircle className="w-4 h-4" />
                {error}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```

---

## FILE: `frontend/src/components/DualViewer.tsx`

```tsx
import React from 'react';

interface DualViewerProps {
  rgbB64: string;
  dsmColorizedB64: string;
  elevationMin: number;
  elevationMax: number;
  unit: string;
}

export default function DualViewer({ rgbB64, dsmColorizedB64, elevationMin, elevationMax, unit }: DualViewerProps) {
  return (
    <div className="glass-panel p-4 space-y-3">
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">2D Comparison</h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <p className="text-xs text-slate-400 text-center">Original RGB</p>
          <img
            src={`data:image/jpeg;base64,${rgbB64}`}
            alt="RGB"
            className="w-full rounded-lg border border-white/5"
          />
        </div>
        <div className="space-y-2">
          <p className="text-xs text-slate-400 text-center">Estimated DSM</p>
          <img
            src={`data:image/png;base64,${dsmColorizedB64}`}
            alt="DSM"
            className="w-full rounded-lg border border-white/5"
          />
        </div>
      </div>
      <div className="flex justify-between text-xs text-slate-500">
        <span>{elevationMin.toFixed(1)} {unit}</span>
        <span className="text-slate-400">Elevation Range</span>
        <span>{elevationMax.toFixed(1)} {unit}</span>
      </div>
    </div>
  );
}
```

---

## FILE: `frontend/src/components/TerrainCanvas.tsx`

```tsx
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
  const velocity = useRef(new THREE.Vector3());
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
```

---

## FILE: `frontend/src/components/FloodSimulator.tsx`

```tsx
import React from 'react';
import { Waves } from 'lucide-react';

interface FloodSimulatorProps {
  waterLevel: number;
  onWaterLevelChange: (level: number) => void;
  elevationMin: number;
  elevationMax: number;
  unit: string;
  dsmRaw: number[][];
}

export default function FloodSimulator({
  waterLevel, onWaterLevelChange, elevationMin, elevationMax, unit, dsmRaw
}: FloodSimulatorProps) {
  // Calculate flooded percentage
  let floodedPercent = 0;
  if (dsmRaw && dsmRaw.length > 0 && waterLevel > 0) {
    let total = 0;
    let flooded = 0;
    for (const row of dsmRaw) {
      for (const val of row) {
        total++;
        if (val < waterLevel) flooded++;
      }
    }
    floodedPercent = total > 0 ? (flooded / total) * 100 : 0;
  }

  return (
    <div className="glass-panel p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Waves className="w-4 h-4 text-blue-400" />
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
          Flood Simulator
        </h3>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Water Level</span>
          <span className="text-blue-400 font-mono">
            {waterLevel.toFixed(1)} {unit}
          </span>
        </div>
        <input
          type="range"
          min={elevationMin}
          max={elevationMax}
          step={(elevationMax - elevationMin) / 200}
          value={waterLevel}
          onChange={(e) => onWaterLevelChange(parseFloat(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-slate-500">
          <span>{elevationMin.toFixed(1)}</span>
          <span>{elevationMax.toFixed(1)}</span>
        </div>
      </div>

      {waterLevel > 0 && (
        <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3 text-center">
          <p className="text-2xl font-bold text-blue-400">{floodedPercent.toFixed(1)}%</p>
          <p className="text-xs text-slate-400">Area Inundated</p>
        </div>
      )}
    </div>
  );
}
```

---

## FILE: `frontend/src/components/CrossSection.tsx`

```tsx
import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Scissors } from 'lucide-react';

interface CrossSectionProps {
  dsmRaw: number[][];
  unit: string;
  active: boolean;
  onToggle: () => void;
}

interface ProfilePoint {
  distance: number;
  elevation: number;
}

export default function CrossSection({ dsmRaw, unit, active, onToggle }: CrossSectionProps) {
  const [pointA, setPointA] = useState<{ row: number; col: number } | null>(null);
  const [pointB, setPointB] = useState<{ row: number; col: number } | null>(null);

  // For the MVP, we'll provide a demo cross-section through the center of the image
  const profileData = useMemo<ProfilePoint[]>(() => {
    if (!dsmRaw || dsmRaw.length === 0) return [];

    const h = dsmRaw.length;
    const w = dsmRaw[0]?.length || 0;
    if (w === 0) return [];

    // Default: horizontal cross-section through middle row
    const midRow = Math.floor(h / 2);
    const K = Math.min(w, 256);
    const points: ProfilePoint[] = [];

    for (let k = 0; k < K; k++) {
      const col = Math.floor((k / (K - 1)) * (w - 1));
      const elevation = dsmRaw[midRow]?.[col] ?? 0;
      const distance = (k / (K - 1)) * w;
      points.push({ distance: Math.round(distance * 10) / 10, elevation: Math.round(elevation * 100) / 100 });
    }

    return points;
  }, [dsmRaw]);

  if (!active || profileData.length === 0) {
    return (
      <div className="glass-panel p-4">
        <button
          onClick={onToggle}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all
            ${active
              ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
              : 'bg-white/5 text-slate-400 hover:bg-white/10 border border-white/5'
            }`}
        >
          <Scissors className="w-4 h-4" />
          Cross-Section
        </button>
      </div>
    );
  }

  return (
    <div className="glass-panel p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Scissors className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Elevation Profile
          </h3>
        </div>
        <button
          onClick={onToggle}
          className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          Close
        </button>
      </div>

      <p className="text-xs text-slate-500">
        Horizontal cross-section through center of image
      </p>

      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={profileData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="distance"
              stroke="#64748b"
              fontSize={10}
              tickFormatter={(v) => `${v.toFixed(0)}`}
              label={{ value: 'Distance (px)', position: 'bottom', fill: '#64748b', fontSize: 10 }}
            />
            <YAxis
              stroke="#64748b"
              fontSize={10}
              tickFormatter={(v) => `${v.toFixed(1)}`}
              label={{ value: `Elevation (${unit})`, angle: -90, position: 'left', fill: '#64748b', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: '#e2e8f0',
                fontSize: '12px',
              }}
              formatter={(value: number) => [`${value.toFixed(2)} ${unit}`, 'Elevation']}
              labelFormatter={(label) => `Distance: ${label} px`}
            />
            <Line
              type="monotone"
              dataKey="elevation"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
```

---

## FILE: `frontend/src/components/ContourOverlay.tsx`

```tsx
import React from 'react';
import { Map } from 'lucide-react';

interface ContourOverlayProps {
  active: boolean;
  onToggle: () => void;
  interval: number;
  onIntervalChange: (val: number) => void;
}

export default function ContourOverlay({ active, onToggle, interval, onIntervalChange }: ContourOverlayProps) {
  return (
    <div className="space-y-2">
      <button
        onClick={onToggle}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all w-full
          ${active
            ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
            : 'bg-white/5 text-slate-400 hover:bg-white/10 border border-white/5'
          }`}
      >
        <Map className="w-4 h-4" />
        Contour Lines
        <span className="ml-auto text-xs opacity-60">{active ? 'ON' : 'OFF'}</span>
      </button>

      {active && (
        <div className="pl-2 space-y-1">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Interval</span>
            <span className="font-mono">{interval.toFixed(1)}</span>
          </div>
          <input
            type="range"
            min={0.5}
            max={20}
            step={0.5}
            value={interval}
            onChange={(e) => onIntervalChange(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      )}
    </div>
  );
}
```

---

## FILE: `frontend/src/components/ColorBar.tsx`

```tsx
import React from 'react';

interface ColorBarProps {
  min: number;
  max: number;
  unit: string;
}

export default function ColorBar({ min, max, unit }: ColorBarProps) {
  return (
    <div className="flex flex-col items-center gap-1">
      <span className="text-xs text-slate-400">{max.toFixed(1)}</span>
      <div
        className="w-4 h-32 rounded-full border border-white/10"
        style={{
          background: 'linear-gradient(to bottom, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff)',
        }}
      />
      <span className="text-xs text-slate-400">{min.toFixed(1)}</span>
      <span className="text-[10px] text-slate-500">{unit}</span>
    </div>
  );
}
```

---

## FILE: `frontend/src/components/SettingsPanel.tsx`

```tsx
import React from 'react';
import { Settings, Mountain, Download } from 'lucide-react';
import ContourOverlay from './ContourOverlay';

interface SettingsPanelProps {
  verticalScale: number;
  onVerticalScaleChange: (val: number) => void;
  showContours: boolean;
  onContoursToggle: () => void;
  contourInterval: number;
  onContourIntervalChange: (val: number) => void;
  requestId: string;
}

export default function SettingsPanel({
  verticalScale, onVerticalScaleChange,
  showContours, onContoursToggle,
  contourInterval, onContourIntervalChange,
  requestId,
}: SettingsPanelProps) {

  const handleExport = async () => {
    try {
      const response = await fetch(`/api/export/${requestId}`);
      if (!response.ok) throw new Error('Export failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `depthwizard_dsm_${requestId}.tif`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export error:', err);
    }
  };

  return (
    <div className="glass-panel p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Settings className="w-4 h-4 text-blue-400" />
        <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Settings</h3>
      </div>

      {/* Vertical Exaggeration */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Mountain className="w-3 h-3" />
          <span>Vertical Scale</span>
          <span className="ml-auto font-mono text-blue-400">{verticalScale.toFixed(1)}×</span>
        </div>
        <input
          type="range"
          min={0.1}
          max={5}
          step={0.1}
          value={verticalScale}
          onChange={(e) => onVerticalScaleChange(parseFloat(e.target.value))}
          className="w-full"
        />
      </div>

      {/* Contour Lines */}
      <ContourOverlay
        active={showContours}
        onToggle={onContoursToggle}
        interval={contourInterval}
        onIntervalChange={onContourIntervalChange}
      />

      {/* Export */}
      <button
        onClick={handleExport}
        className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm w-full
          bg-emerald-500/10 text-emerald-400 border border-emerald-500/20
          hover:bg-emerald-500/20 transition-all"
      >
        <Download className="w-4 h-4" />
        Export GeoTIFF
      </button>
    </div>
  );
}
```

---

## FILE: `frontend/src/App.tsx`

```tsx
import React, { useState } from 'react';
import { useInference, InferenceResult } from './hooks/useInference';
import Dropzone from './components/Dropzone';
import DualViewer from './components/DualViewer';
import TerrainCanvas from './components/TerrainCanvas';
import FloodSimulator from './components/FloodSimulator';
import CrossSection from './components/CrossSection';
import SettingsPanel from './components/SettingsPanel';
import ColorBar from './components/ColorBar';
import { RotateCcw, Clock, Ruler } from 'lucide-react';

export default function App() {
  const { upload, data, loading, error, progress, reset } = useInference();

  // UI state
  const [verticalScale, setVerticalScale] = useState(1.5);
  const [waterLevel, setWaterLevel] = useState(0);
  const [showContours, setShowContours] = useState(false);
  const [contourInterval, setContourInterval] = useState(5);
  const [showCrossSection, setShowCrossSection] = useState(false);

  const handleFileSelected = (file: File) => {
    upload(file);
  };

  // No data yet — show dropzone
  if (!data) {
    return (
      <div className="h-screen flex flex-col">
        {/* Header */}
        <header className="glass-panel-strong mx-4 mt-4 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
              D
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">DepthWizard</h1>
              <p className="text-[10px] text-slate-500 -mt-0.5">SIH26175 · ISRO SAC</p>
            </div>
          </div>
          <span className="text-xs text-slate-500">Single-View Height Estimation & 3D Flythrough</span>
        </header>

        {/* Dropzone */}
        <div className="flex-1">
          <Dropzone onFileSelected={handleFileSelected} loading={loading} progress={progress} />
        </div>

        {error && (
          <div className="mx-4 mb-4 glass-panel px-4 py-3 border border-red-500/20 text-red-400 text-sm">
            ⚠️ {error}
          </div>
        )}
      </div>
    );
  }

  // Data loaded — show results
  const cal = data.calibration;
  const stats = data.mesh_stats;

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="glass-panel-strong mx-2 mt-2 px-4 py-2 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-xs">
            D
          </div>
          <h1 className="text-sm font-bold text-white">DepthWizard</h1>

          {/* Stats badges */}
          <div className="flex gap-2 ml-4">
            <span className="px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 text-[10px] font-mono border border-blue-500/20">
              <Clock className="w-3 h-3 inline mr-1" />
              {data.inference_time_ms.toFixed(0)}ms
            </span>
            <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-mono border border-emerald-500/20">
              <Ruler className="w-3 h-3 inline mr-1" />
              {stats.original_width}×{stats.original_height}
            </span>
            <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-mono border border-purple-500/20">
              {data.is_georef ? '🌐 Georeferenced' : '📷 Relative'}
            </span>
          </div>
        </div>

        <button
          onClick={reset}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/5 text-slate-400 text-xs
            hover:bg-white/10 transition-all border border-white/5"
        >
          <RotateCcw className="w-3 h-3" />
          New Image
        </button>
      </header>

      {/* Main content */}
      <div className="flex-1 flex gap-2 p-2 min-h-0">
        {/* Left sidebar */}
        <div className="w-72 flex flex-col gap-2 overflow-y-auto">
          <DualViewer
            rgbB64={data.rgb_b64}
            dsmColorizedB64={data.dsm_colorized_b64}
            elevationMin={cal.min}
            elevationMax={cal.max}
            unit={cal.unit}
          />

          <FloodSimulator
            waterLevel={waterLevel}
            onWaterLevelChange={setWaterLevel}
            elevationMin={cal.min}
            elevationMax={cal.max}
            unit={cal.unit}
            dsmRaw={data.dsm_raw}
          />

          <CrossSection
            dsmRaw={data.dsm_raw}
            unit={cal.unit}
            active={showCrossSection}
            onToggle={() => setShowCrossSection(!showCrossSection)}
          />

          <SettingsPanel
            verticalScale={verticalScale}
            onVerticalScaleChange={setVerticalScale}
            showContours={showContours}
            onContoursToggle={() => setShowContours(!showContours)}
            contourInterval={contourInterval}
            onContourIntervalChange={setContourInterval}
            requestId={data.request_id}
          />
        </div>

        {/* 3D Canvas */}
        <div className="flex-1 glass-panel overflow-hidden relative">
          <TerrainCanvas
            heightmapB64={data.heightmap_b64}
            rgbB64={data.rgb_b64}
            meshStats={data.mesh_stats}
            verticalScale={verticalScale}
            waterLevel={waterLevel}
            showContours={showContours}
            dsmRaw={data.dsm_raw}
          />
          <div className="absolute top-3 right-3">
            <ColorBar min={cal.min} max={cal.max} unit={cal.unit} />
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## FINAL NOTE TO IMPLEMENTING AGENT

After creating ALL the above files:

1. **Start the backend:**
   ```powershell
   cd E:/rishabh/sih/sih-2026-problem-statements-main` `(1)/sih-2026-problem-statements-main/depthwizard/backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend (new terminal):**
   ```powershell
   cd E:/rishabh/sih/sih-2026-problem-statements-main` `(1)/sih-2026-problem-statements-main/depthwizard/frontend
   npm run dev
   ```

3. **Test:** Open http://localhost:5173, upload any satellite/aerial image.

4. **Verify ALL of these work:**
   - ✅ Image uploads and processes
   - ✅ 2D DualViewer shows RGB vs colored DSM
   - ✅ 3D terrain renders with displacement
   - ✅ WASD + mouse flythrough works
   - ✅ Flood slider turns terrain blue below threshold
   - ✅ Cross-section shows elevation profile chart
   - ✅ Contour lines toggle works
   - ✅ Export downloads a .tif file
   - ✅ "New Image" button resets

If ANY step fails, fix it immediately before moving to the next task. Do NOT skip broken steps.
