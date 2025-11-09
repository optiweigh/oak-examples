import { useEffect, useRef } from "react";
import { useConnection } from "@luxonis/depthai-viewer-common";

export function FacesStatsPoller() {
  const { connected, daiConnection } = useConnection();
  const inFlight = useRef(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!connected) return;

    const tick = () => {
      if (inFlight.current || document.hidden) return;
      inFlight.current = true;

      // @ts-ignore custom service name
      daiConnection?.postToService("Get Faces", {}, (_resp: any) => {
        // console.log("[Get Faces]", _resp); // keep off while debugging streams
        inFlight.current = false;
      });
    };

    const loop = () => {
      tick();
      timer.current = window.setTimeout(loop, 1000); // 1 Hz
    };

    loop();
    return () => {
      if (timer.current) clearTimeout(timer.current);
      inFlight.current = false;
    };
  }, [connected, daiConnection]);

  return null;
}
