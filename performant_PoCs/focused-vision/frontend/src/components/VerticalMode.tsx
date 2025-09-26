import React, { useState } from "react";
import { css } from "../../styled-system/css/css.mjs";
import { Streams } from "@luxonis/depthai-viewer-common";
import { 
  topicGroups, 
  VIDEO_TOPICS_FOCUSED, 
  VIDEO_TOPICS_NON_FOCUSED,
  EYES_MOSAIC_FOCUSED_TOPICS,
  EYES_MOSAIC_NON_FOCUSED_TOPICS,
  defaultOpenAllFocused,
  defaultOpenAllNonFocused
} from "../constants";
import { ArrowDown } from "./ArrowDown";

const PatchedStreams = Streams as unknown as React.ComponentType<any>;

interface VerticalModeProps {
  connected: boolean;
}

export function VerticalMode({ connected }: VerticalModeProps) {
  const [faceDetectionEnabled, setFaceDetectionEnabled] = useState(false);

  return (
    <div
      className={css({
        flex: 1,
        minHeight: 0,
        display: "flex",
        flexDirection: "column",
        gap: "lg",
        overflowY: "auto",
      })}
    >
      {/* Stream 1: Video with face annotations */}
        <section className={css({ minHeight: "500px", display: "flex", flexDirection: "column" })}>
          <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
            {faceDetectionEnabled ? "Original Video with Face Detection" : "Original Video with Direct Eye Detection"}
          </h2>
          <div className={css({ flex: 1, minHeight: "400px", pointerEvents: "none" })}>
            {faceDetectionEnabled ? (
              <PatchedStreams
                key={`video-focused-${connected ? "on" : "off"}`}
                topicGroups={{
                  Video: "A",
                  "Detections Stage 1": "A",
                }}
                defaultTopics={VIDEO_TOPICS_FOCUSED}
                allowedTopics={VIDEO_TOPICS_FOCUSED}
                disableZoom
              />
            ) : (
              <PatchedStreams
                key={`video-non-focused-${connected ? "on" : "off"}`}
                topicGroups={{
                  Video: "A",
                  "Detections Non Focused": "A",
                }}
                defaultTopics={VIDEO_TOPICS_NON_FOCUSED}
                allowedTopics={VIDEO_TOPICS_NON_FOCUSED}
                disableZoom
              />
            )}
          </div>
        </section>

      {/* Toggle Button */}
      <div className={css({ display: "flex", justifyContent: "center", paddingY: "md" })}>
        <button
          onClick={() => setFaceDetectionEnabled(!faceDetectionEnabled)}
          className={css({
            paddingX: "lg",
            paddingY: "md",
            borderRadius: "lg",
            borderWidth: "2px",
            borderColor: faceDetectionEnabled ? "green.500" : "gray.300",
            backgroundColor: faceDetectionEnabled ? "green.50" : "white",
            color: faceDetectionEnabled ? "green.700" : "gray.700",
            _hover: {
              backgroundColor: faceDetectionEnabled ? "green.100" : "gray.100",
              borderColor: faceDetectionEnabled ? "green.600" : "gray.400"
            },
            fontWeight: "600",
            fontSize: "md",
            transition: "all 0.2s ease-in-out",
          })}
        >
          {faceDetectionEnabled ? "✓ Face Detection Enabled" : "Enable Face Detection"}
        </button>
      </div>

      {/* Arrow - only show if face detection is enabled */}
      {faceDetectionEnabled && <ArrowDown label="Face Detection" />}

      {/* Stream 2: Face Mosaic with eye annotations - only show if enabled */}
      {faceDetectionEnabled && (
        <>
          <section className={css({ minHeight: "500px", display: "flex", flexDirection: "column" })}>
            <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
              Face Mosaic with Eye Detection
            </h2>
            <div className={css({ flex: 1, minHeight: "400px", pointerEvents: "none" })}>
              <PatchedStreams
                key={`face-mosaic-${connected ? "on" : "off"}`}
                topicGroups={topicGroups}
                defaultTopics={defaultOpenAllFocused}
                allowedTopics={["Crops Mosaic", "Detections Stage 2 Crops"]}
                hideToolbar
                disableZoom
              />
            </div>
          </section>

          {/* Arrow */}
          <ArrowDown label="Eye Detection" />
        </>
      )}

      {/* Stream 3: Eyes Mosaic without annotations */}
      <section className={css({ minHeight: "500px", display: "flex", flexDirection: "column" })}>
        <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
          {faceDetectionEnabled ? "Eyes Mosaic (Focused)" : "Eyes Mosaic (Direct)"}
        </h2>
        <div className={css({ flex: 1, minHeight: "400px", pointerEvents: "none" })}>
          <PatchedStreams
            key={`eyes-mosaic-${connected ? "on" : "off"}-${faceDetectionEnabled ? "focused" : "non-focused"}`}
            topicGroups={topicGroups}
            defaultTopics={faceDetectionEnabled ? defaultOpenAllFocused : defaultOpenAllNonFocused}
            allowedTopics={faceDetectionEnabled ? EYES_MOSAIC_FOCUSED_TOPICS : EYES_MOSAIC_NON_FOCUSED_TOPICS}
            hideToolbar
            disableZoom
          />
        </div>
      </section>
    </div>
  );
}