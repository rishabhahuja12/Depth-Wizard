/**
 * Pure pooling and palette-quantization for Voxel/Block DSM rendering.
 *
 * Transforms a high-resolution 2D height grid (dsm_raw) into a downsampled
 * grid of discrete box instances with stepped Turbo palette colors.
 * Zero rendering dependencies — purely mathematical and independently testable.
 */

export interface VoxelizeOptions {
  /** Target block resolution along the primary axis (default: 64, range: 24–96) */
  targetResolution?: number;
  /** Number of discrete elevation bands / colors (default: 8, range: 5–12) */
  bandCount?: number;
  /** Optional minimum elevation override (default: min value in dsmRaw) */
  elevationMin?: number;
  /** Optional maximum elevation override (default: max value in dsmRaw) */
  elevationMax?: number;
  /** Gap ratio for X/Z block footprint to create the Minecraft style (default: 0.90, range: 0.85–0.95) */
  gapRatio?: number;
  /** World width of the terrain plane (default: dsmRaw[0].length * 0.1) */
  worldWidth?: number;
  /** World depth of the terrain plane (default: dsmRaw.length * 0.1) */
  worldDepth?: number;
}

export interface VoxelBlock {
  /** Column index in the downsampled grid (0 <= gridX < cols) */
  gridX: number;
  /** Row index in the downsampled grid (0 <= gridZ < rows) */
  gridZ: number;
  /** World-space X center coordinate */
  posX: number;
  /** World-space Z center coordinate */
  posZ: number;
  /** Averaged raw elevation in physical or relative units */
  rawHeight: number;
  /** Normalized elevation in [0, 1] relative to elevation range */
  normalizedHeight: number;
  /** Quantized band index (0 <= bandIndex < bandCount) */
  bandIndex: number;
  /** Hex color string (e.g. "#00bfff") */
  colorHex: string;
  /** Normalized RGB components in [0, 1] for direct use with Three.Color */
  colorRgb: [number, number, number];
}

export interface VoxelGrid {
  /** All voxel block descriptors */
  blocks: VoxelBlock[];
  /** Downsampled grid rows count */
  rows: number;
  /** Downsampled grid columns count */
  cols: number;
  /** Cell physical footprint width */
  cellWidth: number;
  /** Cell physical footprint depth */
  cellDepth: number;
  /** Minimum elevation across the pooled grid */
  elevationMin: number;
  /** Maximum elevation across the pooled grid */
  elevationMax: number;
  /** Elevation range (elevationMax - elevationMin) */
  elevationRange: number;
  /** Discrete Turbo palette hex strings */
  palette: string[];
}

/**
 * Samples a continuous Turbo-like colormap at normalized position t in [0, 1].
 * Matches the blue -> cyan -> green -> yellow -> red progression in mesh_builder.py.
 */
export function sampleTurboRgb(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  let r = 0;
  let g = 0;
  let b = 0;

  if (clamped < 0.25) {
    r = 0.0;
    g = clamped * 4.0;
    b = 1.0;
  } else if (clamped < 0.5) {
    r = 0.0;
    g = 1.0;
    b = 1.0 - (clamped - 0.25) * 4.0;
  } else if (clamped < 0.75) {
    r = (clamped - 0.5) * 4.0;
    g = 1.0;
    b = 0.0;
  } else {
    r = 1.0;
    g = 1.0 - (clamped - 0.75) * 4.0;
    b = 0.0;
  }

  return [r, g, b];
}

/**
 * Converts normalized [r, g, b] in [0, 1] to a 6-character hex string (#rrggbb).
 */
export function rgbToHex(rgb: [number, number, number]): string {
  const toHex = (c: number) => {
    const val = Math.max(0, Math.min(255, Math.round(c * 255)));
    return val.toString(16).padStart(2, '0');
  };
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;
}

/**
 * Generates an array of discrete Turbo palette colors for a specified band count.
 * Each band's color is sampled at the midpoint of its normalized bracket.
 */
export function generateTurboPalette(bandCount: number): string[] {
  const count = Math.max(1, Math.floor(bandCount));
  const palette: string[] = [];
  for (let i = 0; i < count; i++) {
    const t = (i + 0.5) / count;
    palette.push(rgbToHex(sampleTurboRgb(t)));
  }
  return palette;
}

/**
 * Pools a 2D height grid (dsm_raw) into a coarse block grid and quantizes
 * each block into discrete elevation bands with corresponding Turbo palette colors.
 *
 * @param dsmRaw - Row-major 2D array of elevations [row][col]
 * @param options - Configuration for target resolution, bands, footprint scaling, etc.
 * @returns VoxelGrid containing array of block descriptors, grid dimensions, and palette.
 */
export function voxelize(
  dsmRaw: number[][],
  options: VoxelizeOptions = {}
): VoxelGrid {
  if (!dsmRaw || dsmRaw.length === 0 || !dsmRaw[0] || dsmRaw[0].length === 0) {
    return {
      blocks: [],
      rows: 0,
      cols: 0,
      cellWidth: 0,
      cellDepth: 0,
      elevationMin: 0,
      elevationMax: 0,
      elevationRange: 0,
      palette: [],
    };
  }

  const srcRows = dsmRaw.length;
  const srcCols = dsmRaw[0].length;

  const targetRes = Math.max(4, Math.min(256, options.targetResolution ?? 64));
  const bandCount = Math.max(2, Math.min(32, options.bandCount ?? 8));
  const gapRatio = Math.max(0.5, Math.min(1.0, options.gapRatio ?? 0.90));

  // Preserve aspect ratio when determining target rows and columns
  let cols: number;
  let rows: number;
  if (srcCols >= srcRows) {
    cols = targetRes;
    rows = Math.max(1, Math.round(targetRes * (srcRows / srcCols)));
  } else {
    rows = targetRes;
    cols = Math.max(1, Math.round(targetRes * (srcCols / srcRows)));
  }

  // World dimensions matching Three.js PlaneGeometry(w * 0.1, h * 0.1)
  const worldW = options.worldWidth ?? srcCols * 0.1;
  const worldD = options.worldDepth ?? srcRows * 0.1;

  const cellWidth = (worldW / cols) * gapRatio;
  const cellDepth = (worldD / rows) * gapRatio;

  const palette = generateTurboPalette(bandCount);

  // First pass: pool elevations using area-averaging
  const pooledGrid: number[][] = [];
  let foundMin = Infinity;
  let foundMax = -Infinity;

  for (let r = 0; r < rows; r++) {
    const rowVals: number[] = [];
    const srcR0 = Math.floor((r * srcRows) / rows);
    const srcR1 = Math.max(srcR0 + 1, Math.floor(((r + 1) * srcRows) / rows));

    for (let c = 0; c < cols; c++) {
      const srcC0 = Math.floor((c * srcCols) / cols);
      const srcC1 = Math.max(srcC0 + 1, Math.floor(((c + 1) * srcCols) / cols));

      let sum = 0;
      let count = 0;

      for (let y = srcR0; y < srcR1; y++) {
        const srcRow = dsmRaw[y];
        if (!srcRow) continue;
        for (let x = srcC0; x < srcC1; x++) {
          const val = srcRow[x];
          if (val !== undefined && !Number.isNaN(val) && Number.isFinite(val)) {
            sum += val;
            count++;
          }
        }
      }

      const avg = count > 0 ? sum / count : 0;
      rowVals.push(avg);
      if (avg < foundMin) foundMin = avg;
      if (avg > foundMax) foundMax = avg;
    }
    pooledGrid.push(rowVals);
  }

  if (foundMin === Infinity) foundMin = 0;
  if (foundMax === -Infinity) foundMax = 1;

  // Use provided min/max overrides if specified
  const elevMin = options.elevationMin ?? foundMin;
  const elevMax = options.elevationMax ?? foundMax;
  const elevRange = Math.max(1e-6, elevMax - elevMin);

  // Second pass: construct VoxelBlocks with world coordinates and quantized colors
  const blocks: VoxelBlock[] = [];
  const halfWorldW = worldW / 2;
  const halfWorldD = worldD / 2;
  const stepX = worldW / cols;
  const stepZ = worldD / rows;

  for (let r = 0; r < rows; r++) {
    const posZ = -halfWorldD + (r + 0.5) * stepZ;
    for (let c = 0; c < cols; c++) {
      const rawH = pooledGrid[r][c];
      const normH = Math.max(0, Math.min(1, (rawH - elevMin) / elevRange));
      const bandIndex = Math.min(bandCount - 1, Math.floor(normH * bandCount));
      const colorHex = palette[bandIndex] ?? palette[0];
      const colorRgb = sampleTurboRgb((bandIndex + 0.5) / bandCount);
      const posX = -halfWorldW + (c + 0.5) * stepX;

      blocks.push({
        gridX: c,
        gridZ: r,
        posX,
        posZ,
        rawHeight: rawH,
        normalizedHeight: normH,
        bandIndex,
        colorHex,
        colorRgb,
      });
    }
  }

  return {
    blocks,
    rows,
    cols,
    cellWidth,
    cellDepth,
    elevationMin: elevMin,
    elevationMax: elevMax,
    elevationRange: elevRange,
    palette,
  };
}

/**
 * Computes quantized world-space surface height for a voxel block (§3),
 * matching VoxelMesh instance height in VoxelTerrain.tsx exactly.
 */
export function quantizeVoxelHeightWorld(
  rawH: number,
  elevationMin: number,
  elevationRange: number,
  verticalScale: number = 1.0,
  isRelative: boolean = false
): number {
  const normH = Math.max(0, Math.min(1, (rawH - elevationMin) / Math.max(1e-6, elevationRange)));
  return isRelative
    ? Math.max(0.12, normH * 18.0 * verticalScale)
    : Math.max(0.12, (rawH - elevationMin) * verticalScale * 0.1);
}
