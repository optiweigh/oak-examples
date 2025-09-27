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
  defaultOpenAllNonFocused,
  NN_INPUT_FACE_TOPICS,
  NN_INPUT_EYE_TOPICS
} from "../constants";
import { ZigZagArrow } from "./ZigZagArrow";

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
        padding: "lg",
      })}
    >
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

      {/* Zig-Zag Layout Container */}
      <div className={css({ display: "flex", flexDirection: "column", gap: "xl" })}>
        
        {/* Row 1: Video (Left) */}
        <div className={css({ display: "flex", justifyContent: "flex-start" })}>
          <section className={css({ 
            width: "60%", 
            minHeight: "600px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
          })}>
            <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
              {faceDetectionEnabled ? "Original Video with Face Detection" : "Original Video with Direct Eye Detection"}
            </h2>
            <div className={css({ flex: 1, minHeight: "500px", pointerEvents: "none" })}>
              {faceDetectionEnabled ? (
                <PatchedStreams
                  key={`video-focused-${connected ? "on" : "off"}`}
                  topicGroups={{
                    Video: "A",
                    "Full Frame eyes detection": "A",
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
        </div>

        {/* Arrow from Video to NN Input */}
        <ZigZagArrow 
          label={faceDetectionEnabled ? "Face Detection" : "Direct Eye Detection"} 
          direction="down-right" 
        />

        {/* Row 2: NN Inpots */}
        <div className={css({ display: "flex", justifyContent: "flex-end" })}>
          <section className={css({ 
            width: "60%", 
            minHeight: "500px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
          })}>
            <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
              {faceDetectionEnabled ? "NN input Face Detection" : "NN input Eye Detection"}
            </h2>
            <div className={css({ flex: 1, minHeight: "400px", pointerEvents: "none" })}>
              <PatchedStreams
                key={`eyes-mosaic-${connected ? "on" : "off"}-${faceDetectionEnabled ? "focused" : "non-focused"}`}
                topicGroups={faceDetectionEnabled ? {"NN input Face Detection": "A",
                  "Detections Stage 1": "A"} : {"NN input Eye Detection": "A",
                    "Detections Non Focused": "A"}}
                defaultTopics={faceDetectionEnabled ? defaultOpenAllFocused : defaultOpenAllNonFocused}
                allowedTopics={faceDetectionEnabled ? NN_INPUT_FACE_TOPICS : NN_INPUT_EYE_TOPICS}
                hideToolbar
                disableZoom
              />
            </div>
          </section>
        </div>

        {/* Arrow from NN Input to Face Mosaic (only if face detection enabled) */}
        {faceDetectionEnabled && (
          <ZigZagArrow 
            label="Face Cropping" 
            direction="left" 
          />
        )}

        {/* Row Optional: Face Mosaic (Right) - only show if enabled */}
        {faceDetectionEnabled && (
          <div className={css({ display: "flex", justifyContent: "flex-start" })}>
            <section className={css({ 
              width: "60%", 
              minHeight: "500px", 
              display: "flex", 
              flexDirection: "column",
              backgroundColor: "white",
              borderRadius: "lg",
              padding: "md",
              boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            })}>
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
          </div>
        )}

        {/* Arrow from Face Mosaic to Eyes Mosaic (only if face detection enabled) */}
        {faceDetectionEnabled && (
          <ZigZagArrow 
            label="Eye Detection" 
            direction="down-right" 
          />
        )}

        <ZigZagArrow 
          label={faceDetectionEnabled ? "Eye Detection" : "Eyes Mosaic"} 
          direction= {faceDetectionEnabled ? "down-right" : "left" }
        />

        {/* Row 3: Eyes Mosaic (Left) */}
        <div className={css({ display: "flex", justifyContent: faceDetectionEnabled ? "flex-end" : "flex-start" })}>
          <section className={css({ 
            width: "60%", 
            minHeight: "500px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
          })}>
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

      </div>
    </div>
  );
}