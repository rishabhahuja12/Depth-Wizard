import React, { forwardRef } from 'react';
import * as THREE from 'three';

export interface DroneModelProps {
  throttle?: number;
}

/**
 * Procedural Drone Mesh rig scaffold.
 */
export const DroneModel = forwardRef<THREE.Group, DroneModelProps>(function DroneModel(
  { throttle = 0 },
  ref
) {
  return (
    <group ref={ref}>
      {/* Central body fuselage scaffold */}
      <mesh castShadow receiveShadow position={[0, 0, 0]}>
        <boxGeometry args={[0.5, 0.15, 0.7]} />
        <meshStandardMaterial color="#1a1d24" roughness={0.3} metalness={0.8} />
      </mesh>
    </group>
  );
});

export default DroneModel;
