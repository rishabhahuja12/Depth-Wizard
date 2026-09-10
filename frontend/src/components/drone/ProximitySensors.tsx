import React, { forwardRef } from 'react';
import * as THREE from 'three';

export interface SensorReadings {
  left: number;
  right: number;
  bottom: number;
}

export interface ProximitySensorsProps {
  droneRef: React.RefObject<THREE.Group>;
  dsmRaw: number[][];
  meshStats: {
    width: number;
    height: number;
    elevation_min: number;
    elevation_max: number;
  };
  onReadingsUpdate?: (readings: SensorReadings) => void;
}

/**
 * ProximitySensors scaffold.
 */
export const ProximitySensors = forwardRef<THREE.Group, ProximitySensorsProps>(
  function ProximitySensors(_props, ref) {
    return <group ref={ref} />;
  }
);

export default ProximitySensors;
