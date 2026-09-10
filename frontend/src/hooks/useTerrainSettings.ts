import { useState } from 'react';

export interface TerrainSettings {
  verticalScale: number;
  setVerticalScale: React.Dispatch<React.SetStateAction<number>>;
  waterLevel: number;
  setWaterLevel: React.Dispatch<React.SetStateAction<number>>;
  showContours: boolean;
  setShowContours: React.Dispatch<React.SetStateAction<boolean>>;
  contourInterval: number;
  setContourInterval: React.Dispatch<React.SetStateAction<number>>;
  renderMode: 'voxel' | 'smooth';
  setRenderMode: React.Dispatch<React.SetStateAction<'voxel' | 'smooth'>>;
  voxelBands: number;
  setVoxelBands: React.Dispatch<React.SetStateAction<number>>;
  voxelResolution: number;
  setVoxelResolution: React.Dispatch<React.SetStateAction<number>>;
}

export interface TerrainSettingsInitialValues {
  initialVerticalScale?: number;
  initialWaterLevel?: number;
  initialShowContours?: boolean;
  initialContourInterval?: number;
  initialRenderMode?: 'voxel' | 'smooth';
  initialVoxelBands?: number;
  initialVoxelResolution?: number;
}

/**
 * useTerrainSettings (§5):
 * Unified single source of truth for terrain and voxel configuration.
 * Shared between the 01 SURFACE sidebar tab and the floating TerrainQuickPanel.
 */
export function useTerrainSettings(initialValues?: TerrainSettingsInitialValues): TerrainSettings {
  const [verticalScale, setVerticalScale] = useState<number>(initialValues?.initialVerticalScale ?? 1.5);
  const [waterLevel, setWaterLevel] = useState<number>(initialValues?.initialWaterLevel ?? 0);
  const [showContours, setShowContours] = useState<boolean>(initialValues?.initialShowContours ?? false);
  const [contourInterval, setContourInterval] = useState<number>(initialValues?.initialContourInterval ?? 5);
  const [renderMode, setRenderMode] = useState<'voxel' | 'smooth'>(initialValues?.initialRenderMode ?? 'voxel');
  const [voxelBands, setVoxelBands] = useState<number>(initialValues?.initialVoxelBands ?? 8);
  const [voxelResolution, setVoxelResolution] = useState<number>(initialValues?.initialVoxelResolution ?? 64);

  return {
    verticalScale,
    setVerticalScale,
    waterLevel,
    setWaterLevel,
    showContours,
    setShowContours,
    contourInterval,
    setContourInterval,
    renderMode,
    setRenderMode,
    voxelBands,
    setVoxelBands,
    voxelResolution,
    setVoxelResolution,
  };
}
