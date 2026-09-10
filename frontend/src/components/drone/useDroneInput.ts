import { useRef, useEffect } from 'react';

export interface DroneInputState {
  pitchForward: boolean; // W / ArrowUp
  pitchBackward: boolean; // S / ArrowDown
  yawLeft: boolean; // A / ArrowLeft
  yawRight: boolean; // D / ArrowRight
  throttleUp: boolean; // Space / Q
  throttleDown: boolean; // E
  turbo: boolean; // Shift
}

export function useDroneInput(active: boolean = true) {
  const inputRef = useRef<DroneInputState>({
    pitchForward: false,
    pitchBackward: false,
    yawLeft: false,
    yawRight: false,
    throttleUp: false,
    throttleDown: false,
    turbo: false,
  });

  useEffect(() => {
    if (!active) {
      inputRef.current = {
        pitchForward: false,
        pitchBackward: false,
        yawLeft: false,
        yawRight: false,
        throttleUp: false,
        throttleDown: false,
        turbo: false,
      };
      return;
    }

    const onKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === 'w' || key === 'arrowup') inputRef.current.pitchForward = true;
      if (key === 's' || key === 'arrowdown') inputRef.current.pitchBackward = true;
      if (key === 'a' || key === 'arrowleft') inputRef.current.yawLeft = true;
      if (key === 'd' || key === 'arrowright') inputRef.current.yawRight = true;
      if (key === ' ' || key === 'q') inputRef.current.throttleUp = true;
      if (key === 'e') inputRef.current.throttleDown = true;
      if (key === 'shift') inputRef.current.turbo = true;
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === 'w' || key === 'arrowup') inputRef.current.pitchForward = false;
      if (key === 's' || key === 'arrowdown') inputRef.current.pitchBackward = false;
      if (key === 'a' || key === 'arrowleft') inputRef.current.yawLeft = false;
      if (key === 'd' || key === 'arrowright') inputRef.current.yawRight = false;
      if (key === ' ' || key === 'q') inputRef.current.throttleUp = false;
      if (key === 'e') inputRef.current.throttleDown = false;
      if (key === 'shift') inputRef.current.turbo = false;
    };

    const onBlur = () => {
      inputRef.current = {
        pitchForward: false,
        pitchBackward: false,
        yawLeft: false,
        yawRight: false,
        throttleUp: false,
        throttleDown: false,
        turbo: false,
      };
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, [active]);

  return inputRef;
}
