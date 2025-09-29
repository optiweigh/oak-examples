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
  NN_INPUT_EYE_TOPICS,
  FACE_MOSAIC_TOPICS
} from "../constants";
import { ZigZagArrow } from "./ZigZagArrow";

const PatchedStreams = Streams as unknown as React.ComponentType<any>;

interface VerticalModeProps {
  connected: boolean;
  faceDetectionEnabled: boolean;
}

export function VerticalMode({ connected, faceDetectionEnabled }: VerticalModeProps) {
  const [showZigZag, setShowZigZag] = useState(false);

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

    {/* Two Video Streams Side by Side - hide if zig-zag is enabled */}
    {!showZigZag && (
      <>
        <div className={css({ 
          display: "flex", 
          gap: "lg", 
          marginBottom: "xl",
          minHeight: "400px"
        })}>
          {/* Video Stream 1 */}
          <section className={css({ 
            width: "50%", 
            minHeight: "600px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            position: "relative",
          })}>
            <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
              {faceDetectionEnabled ? "Original Video with Face Detection" : "Original Video with Direct Eye Detection"}
            </h2>
            <div className={css({ flex: 1, minHeight: "500px", pointerEvents: "none" })}>
                <PatchedStreams
                  key={`video-focused-${connected ? "on" : "off"}`}
                  topicGroups={faceDetectionEnabled ? {
                    Video: "A",
                    "Full Frame eyes detection": "A",
                  } : {
                    Video: "A",
                    "Detections Non Focused Remapped": "A",
                  }}
                  defaultTopics={faceDetectionEnabled ? VIDEO_TOPICS_FOCUSED : VIDEO_TOPICS_NON_FOCUSED}
                  allowedTopics={faceDetectionEnabled ? VIDEO_TOPICS_FOCUSED : VIDEO_TOPICS_NON_FOCUSED}
                  hideToolbar
                  disableZoom
                />
            </div>
          </section>

          <section className={css({ 
            width: "50%", 
            minHeight: "600px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            position: "relative",
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

         {/* Toggle Button - positioned below streams */}
         <div className={css({ display: "flex", justifyContent: "center", paddingTop: "150px", paddingBottom: "md" })}>
          <button
            onClick={() => setShowZigZag(!showZigZag)}
            className={css({
              paddingX: "lg",
              paddingY: "md",
              borderRadius: "lg",
              borderWidth: "2px",
              borderColor: "gray.300",
              backgroundColor: "white",
              color: "gray.700",
              _hover: {
                backgroundColor: "gray.100",
                borderColor: "gray.400"
              },
              fontWeight: "600",
              fontSize: "md",
              transition: "all 0.2s ease-in-out",
            })}
          >
            Show Pipeline Flow
          </button>
        </div>
      </>
    )}

    {/* Toggle Button - visible in pipeline mode */}
    {showZigZag && (
      <div className={css({ display: "flex", justifyContent: "center", paddingY: "md" })}>
        <button
          onClick={() => setShowZigZag(!showZigZag)}
          className={css({
            paddingX: "lg",
            paddingY: "md",
            borderRadius: "lg",
            borderWidth: "2px",
            borderColor: "blue.500",
            backgroundColor: "blue.50",
            color: "blue.700",
            _hover: {
              backgroundColor: "blue.100",
              borderColor: "blue.600"
            },
            fontWeight: "600",
            fontSize: "md",
            transition: "all 0.2s ease-in-out",
          })}
        >
          ✓ Show Pipeline Flow
        </button>
      </div>
    )}

      {/* Zig-Zag Layout Container - only show if toggled */}
      {showZigZag && (
      <div className={css({ display: "flex", flexDirection: "column", gap: "xl" })}>
        
        {/* Row 1: Video (Left) */}
        <div className={css({ display: "flex", justifyContent: "flex-start", position: "relative" })}>
          <section className={css({ 
            width: "60%", 
            minHeight: "600px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            position: "relative",
          })}>
            <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
              {faceDetectionEnabled ? "Original Video with Face Detection" : "Original Video with Direct Eye Detection"}
            </h2>
            <div className={css({ flex: 1, minHeight: "500px", pointerEvents: "none" })}>
                <PatchedStreams
                  key={`video-focused-${connected ? "on" : "off"}`}
                  topicGroups={faceDetectionEnabled ? {
                    Video: "A",
                    "Full Frame eyes detection": "A",
                  } : {
                    Video: "A",
                    "Detections Non Focused Remapped": "A",
                  }}
                  defaultTopics={faceDetectionEnabled ? VIDEO_TOPICS_FOCUSED : VIDEO_TOPICS_NON_FOCUSED}
                  allowedTopics={faceDetectionEnabled ? VIDEO_TOPICS_FOCUSED : VIDEO_TOPICS_NON_FOCUSED}
                  hideToolbar
                  disableZoom
                />
            </div>
          </section>
          
          {/* Arrow from Video to NN Input */}
          <ZigZagArrow 
            label={faceDetectionEnabled ? "Face Detection" : "Direct Eye Detection"} 
            direction="right" 
          />
        </div>

        {/* Row 2: NN Inpots */}
        <div className={css({ display: "flex", justifyContent: "flex-end", position: "relative" })}>
          <section className={css({ 
            width: "60%", 
            minHeight: "600px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            position: "relative",
          })}>
            <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
              {faceDetectionEnabled ? "NN input Face Detection" : "NN input Eye Detection"}
            </h2>
            <div className={css({ flex: 1, minHeight: "400px", pointerEvents: "none" })}>
              <PatchedStreams
                key={`eyes-mosaic-${connected ? "on" : "off"}-${faceDetectionEnabled ? "focused" : "non-focused"}`}
                topicGroups={faceDetectionEnabled ? {"NN input Face Detection": "A",
                  "Detections Stage 1": "A"} : {"NN input Eye Detection": "A",
                    "Detections NN Non Focused": "A"}}
                defaultTopics={faceDetectionEnabled ? NN_INPUT_FACE_TOPICS : NN_INPUT_EYE_TOPICS}
                allowedTopics={faceDetectionEnabled ? NN_INPUT_FACE_TOPICS : NN_INPUT_EYE_TOPICS}
                hideToolbar
                disableZoom
              />
            </div>
          </section>
          
          {/* Arrow from NN Input to Face Mosaic (only if face detection enabled) */}
          <ZigZagArrow 
            label={faceDetectionEnabled ? "Face Cropping" : "Eyes Cropping"} 
            direction="left" 
          />
        </div>

        {/* Row Optional: Face Mosaic (Right) - only show if enabled */}
        {faceDetectionEnabled && (
          <div className={css({ display: "flex", justifyContent: "flex-start", position: "relative" })}>
            <section className={css({ 
              width: "60%", 
              minHeight: "600px", 
              display: "flex", 
              flexDirection: "column",
              backgroundColor: "white",
              borderRadius: "lg",
              padding: "md",
              boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
              position: "relative",
            })}>
              <h2 className={css({ fontSize: "lg", fontWeight: "700", marginBottom: "sm" })}>
                Face Mosaic with Eye Detection
              </h2>
              <div className={css({ flex: 1, minHeight: "400px", pointerEvents: "none" })}>
                <PatchedStreams
                  key={`face-mosaic-${connected ? "on" : "off"}`}
                  topicGroups={topicGroups}
                  defaultTopics={FACE_MOSAIC_TOPICS}
                  allowedTopics={FACE_MOSAIC_TOPICS}
                  hideToolbar
                  disableZoom
                />
              </div>
            </section>
            
            {/* Arrow from Face Mosaic to Eyes Mosaic */}
            <ZigZagArrow 
              label="Eye Detection" 
              direction="right" 
            />
          </div>
        )}

        {/* Row 3: Eyes Mosaic (Left) */}
        <div className={css({ display: "flex", justifyContent: faceDetectionEnabled ? "flex-end" : "flex-start", position: "relative" })}>
          <section className={css({ 
            width: "60%", 
            minHeight: "600px", 
            display: "flex", 
            flexDirection: "column",
            backgroundColor: "white",
            borderRadius: "lg",
            padding: "md",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
            position: "relative",
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
      )}
    </div>
  );
}