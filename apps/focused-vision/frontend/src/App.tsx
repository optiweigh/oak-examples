// src/App.tsx

import {Streams} from "@luxonis/depthai-viewer-common";

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Focused Vision</h1>
        <p className="caption">
          This dashboard shows three synchronized vision streams.
          The left stream is the low-resolution camera feed with face detections.
          The other two show head-crop outputs from non-focused and focused stages.
        </p>
      </header>

      <main className="streams-row">
        {/* 1) low-res with face detections */}
        <Streams
            defaultTopics={["low_res_image", "face detections"]}
            allowedTopics={["low_res_image", "face detections"]}
            hideToolbar
        />
      </main>
    </div>
  );
};
