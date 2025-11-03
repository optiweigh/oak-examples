import { useState } from "react";
import { css } from "../styled-system/css/css.mjs";
import { useConnection } from "@luxonis/depthai-viewer-common";
import { Header, VerticalMode } from "./components";

export default function App() {
  const { connected } = useConnection();
  const [faceDetectionEnabled, setFaceDetectionEnabled] = useState(false);

  const handleFaceDetectionToggle = () => {
    setFaceDetectionEnabled(!faceDetectionEnabled);
  };

  return (
    <main
      className={css({
        width: "screen",
        height: "screen",
        display: "flex",
        flexDirection: "column",
        gap: "md",
        padding: "md",
      })}
    >
      <Header 
        faceDetectionEnabled={faceDetectionEnabled}
        onFaceDetectionToggle={handleFaceDetectionToggle}
      />
      <VerticalMode 
        connected={connected} 
        faceDetectionEnabled={faceDetectionEnabled}
      />
    </main>
  );
}