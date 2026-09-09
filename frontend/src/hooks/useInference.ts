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
  normal_map_b64?: string;
  dsm_colorized_b64: string;
  mesh_stats: MeshStats;
  calibration: CalibrationData;
  dsm_raw: number[][];
  is_georef: boolean;
  confidence_mean?: number;
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
        const err = await response.json().catch(() => ({ detail: `Server error: ${response.status}` }));
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
