export interface MeshElevationStats {
  width: number;
  height: number;
  elevation_min: number;
  elevation_max: number;
  elevation_range: number;
}

export interface CollisionResult {
  position: { x: number; y: number; z: number };
  velocity: { x: number; y: number; z: number };
  collidedFloor: boolean;
  collidedWall: boolean;
  groundWorldY: number;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

/**
 * Samples the 3D world Y elevation of the terrain at Three.js coordinates (x, z).
 * Reuses the row-major dsm_raw elevation grid and matches the displacementScale
 * of DroneTerrainMesh and TerrainCanvas without expensive mesh raycasting.
 *
 * @param x World X coordinate in Three.js space
 * @param z World Z coordinate in Three.js space
 * @param dsmRaw Row-major 2D elevation grid number[][]
 * @param meshStats DSM dimensions and elevation boundaries
 * @param verticalScale Exaggeration multiplier (default 1.0)
 * @param waterLevel Optional flood plane elevation
 * @returns World Y coordinate of the terrain surface
 */
export function getTerrainElevationAt(
  x: number,
  z: number,
  dsmRaw: number[][],
  meshStats: MeshElevationStats,
  verticalScale = 1.0,
  waterLevel?: number
): number {
  if (!dsmRaw || dsmRaw.length === 0 || !dsmRaw[0] || dsmRaw[0].length === 0) {
    return 0;
  }

  const rows = dsmRaw.length;
  const cols = dsmRaw[0].length;

  // World dimensions matching Three.js PlaneGeometry(meshStats.width * 0.1, meshStats.height * 0.1)
  const worldW = Math.max(1e-4, meshStats.width * 0.1);
  const worldD = Math.max(1e-4, meshStats.height * 0.1);

  // Plane is centered at (0, 0, 0) in world space
  // Normalized UV coordinates [0, 1]
  const u = clamp(x / worldW + 0.5, 0, 1);
  const v = clamp(z / worldD + 0.5, 0, 1);

  // Map to continuous grid indices
  const colFloat = u * (cols - 1);
  const rowFloat = v * (rows - 1);

  const c0 = Math.floor(colFloat);
  const c1 = Math.min(cols - 1, c0 + 1);
  const r0 = Math.floor(rowFloat);
  const r1 = Math.min(rows - 1, r0 + 1);

  const tx = colFloat - c0;
  const ty = rowFloat - r0;

  const h00 = dsmRaw[r0]?.[c0] ?? meshStats.elevation_min;
  const h10 = dsmRaw[r0]?.[c1] ?? h00;
  const h01 = dsmRaw[r1]?.[c0] ?? h00;
  const h11 = dsmRaw[r1]?.[c1] ?? h10;

  // Bilinear interpolation
  const hTop = h00 * (1 - tx) + h10 * tx;
  const hBottom = h01 * (1 - tx) + h11 * tx;
  const rawElev = hTop * (1 - ty) + hBottom * ty;

  // Convert raw elevation to 3D world Y matching DroneTerrainMesh.tsx
  const isRelative = meshStats.elevation_range <= 2.0;
  const displacementScale = isRelative
    ? 18.0 * verticalScale
    : meshStats.elevation_range * verticalScale * 0.1;
  const verticalFactor = isRelative
    ? 18.0 * verticalScale
    : verticalScale * 0.1;

  let worldY: number;
  if (isRelative) {
    const normH = Math.max(0, (rawElev - meshStats.elevation_min) / Math.max(1e-6, meshStats.elevation_range));
    worldY = normH * displacementScale;
  } else {
    worldY = (rawElev - meshStats.elevation_min) * verticalScale * 0.1;
  }

  // Account for flood / water plane
  if (waterLevel !== undefined && waterLevel > meshStats.elevation_min) {
    const waterWorldY = (waterLevel - meshStats.elevation_min) * verticalFactor;
    worldY = Math.max(worldY, waterWorldY);
  }

  return worldY;
}

/**
 * Computes Above Ground Level (AGL) clearance distance in meters.
 */
export function computeAgl(
  droneWorldY: number,
  groundWorldY: number,
  meshStats: MeshElevationStats,
  verticalScale = 1.0
): number {
  const deltaWorldY = Math.max(0, droneWorldY - groundWorldY);
  const isRelative = meshStats.elevation_range <= 2.0;
  if (isRelative) {
    const factor = Math.max(1e-6, 18.0 * verticalScale);
    return (deltaWorldY / factor) * meshStats.elevation_range;
  }
  const factor = Math.max(1e-6, verticalScale * 0.1);
  return deltaWorldY / factor;
}

/**
 * Computes Mean Sea Level (MSL) altitude in meters.
 */
export function computeMsl(
  droneWorldY: number,
  meshStats: MeshElevationStats,
  verticalScale = 1.0
): number {
  const isRelative = meshStats.elevation_range <= 2.0;
  if (isRelative) {
    const factor = Math.max(1e-6, 18.0 * verticalScale);
    return meshStats.elevation_min + (droneWorldY / factor) * meshStats.elevation_range;
  }
  const factor = Math.max(1e-6, verticalScale * 0.1);
  return meshStats.elevation_min + droneWorldY / factor;
}

/**
 * Resolves terrain clearance and collision against dsm_raw (§5):
 * Enforces hard floor clearance above terrain/rooftops (pos.y >= groundY + minClearance)
 * and halts horizontal velocity when impacting steep cliff faces or building walls.
 */
export function resolveTerrainCollision(
  prevPos: { x: number; y: number; z: number },
  candidatePos: { x: number; y: number; z: number },
  velocity: { x: number; y: number; z: number },
  dsmRaw: number[][],
  meshStats: MeshElevationStats,
  verticalScale = 1.0,
  waterLevel?: number,
  minClearance = 1.2
): CollisionResult {
  const targetGroundY = getTerrainElevationAt(
    candidatePos.x,
    candidatePos.z,
    dsmRaw,
    meshStats,
    verticalScale,
    waterLevel
  );

  let collidedFloor = false;
  let collidedWall = false;
  const newPos = { ...candidatePos };
  const newVel = { ...velocity };

  // 1. Horizontal obstacle / wall collision check:
  // If target ground elevation is significantly higher than the drone's previous altitude,
  // the drone has impacted a vertical wall, cliff, or tall building structure.
  if (targetGroundY > prevPos.y + 0.15) {
    collidedWall = true;
    newPos.x = prevPos.x;
    newPos.z = prevPos.z;
    newVel.x *= 0.05;
    newVel.z *= 0.05;

    // Recalculate ground at preserved position
    const currentGroundY = getTerrainElevationAt(
      newPos.x,
      newPos.z,
      dsmRaw,
      meshStats,
      verticalScale,
      waterLevel
    );
    const minAllowedY = currentGroundY + minClearance;
    if (newPos.y < minAllowedY) {
      newPos.y = minAllowedY;
      if (newVel.y < 0) newVel.y = 0;
      collidedFloor = true;
    }
    return {
      position: newPos,
      velocity: newVel,
      collidedFloor,
      collidedWall,
      groundWorldY: currentGroundY,
    };
  }

  // 2. Vertical hard floor & rooftop clearance:
  const minAllowedY = targetGroundY + minClearance;
  if (newPos.y < minAllowedY) {
    newPos.y = minAllowedY;
    if (newVel.y < 0) newVel.y = 0;
    collidedFloor = true;
  }

  return {
    position: newPos,
    velocity: newVel,
    collidedFloor,
    collidedWall,
    groundWorldY: targetGroundY,
  };
}
