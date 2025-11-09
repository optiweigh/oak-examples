import { useEffect } from "react";
import { useConnection } from "@luxonis/depthai-viewer-common";

export function FacesStatsPoller() {
  const connection = useConnection();

  useEffect(() => {
    if (!connection.connected) return;

    const tick = () => {
      // @ts-ignore custom service name
      connection.daiConnection?.postToService("Get Faces", {}, (resp: any) => {
        console.log("[Get Faces] payload:", resp);
        if (resp?.stats) {
          console.log(
            `[stats] avgAge=${resp.stats.age} males=${resp.stats.males}% females=${resp.stats.females}%`,
            `emotions: happy=${resp.stats.happy}% neutral=${resp.stats.neutral}%`,
            `surprise=${resp.stats.surprise}% angry=${resp.stats.angry}% sad=${resp.stats.sad}%`
          );
        }
      });
    };

    const id = setInterval(tick, 1000);
    tick();
    return () => clearInterval(id);
  }, [connection.connected, connection.daiConnection]);

  return null;
}
