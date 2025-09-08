import React, { useMemo, useState } from "react";
import { css } from "../styled-system/css/css.mjs";
import { Streams, useConnection } from "@luxonis/depthai-viewer-common";

const PatchedStreams = Streams as unknown as React.ComponentType<any>;

const topicGroups = {
  "Video": "A",
  "Detections Stage 1": "A",
  "Crops Mosaic": "B",
  "Detections Stage 2 Crops": "B",
  "Non Focused Video": "C",
  "Detections Non Focused": "C",
};

// presets
const FOCUSED_TOPICS = ["Crops Mosaic", "Video"];
const NON_FOCUSED_TOPICS = ["Non Focused Video"];
const defaultOpenAll    = ["Video", "Crops Mosaic", "Non Focused Video"];

const ORDER_AB = ["Video", "Crops Mosaic"];
const sortAB = (a: { name: string }, b: { name: string }) => {
  const ai = ORDER_AB.indexOf(a.name);
  const bi = ORDER_AB.indexOf(b.name);
  return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
};

export default function App() {
  type LayoutMode = "split" | "single";
  const [layout, setLayout] = useState<LayoutMode>("split");
  const [mode, setMode] = useState<"focused" | "nonFocused">("focused");
  const { connected } = useConnection();

  const defaultTopics = useMemo(
    () => (mode === "focused" ? FOCUSED_TOPICS : NON_FOCUSED_TOPICS),
    [mode]
  );

  const headerRight = (
    <button
      onClick={() => setLayout(l => (l === "split" ? "single" : "split"))}
      className={css({
        paddingX: "md",
        paddingY: "xs",
        borderWidth: "1px",
        borderColor: "gray.300",
        borderRadius: "md",
        backgroundColor: "transparent",
        _hover: { backgroundColor: "gray.100" },
        fontWeight: "600",
      })}
    >
      {layout === "split" ? "Switch mode" : "Back to 3 streams"}
    </button>
  );

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
      {/* Header */}
      <header
        className={css({
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        })}
      >
        <h1 className={css({ fontSize: "xl", fontWeight: "600" })}>
          {layout === "split"
            ? "Focused & Non Focused"
            : mode === "focused"
              ? "Focused mode"
              : "Non-focused mode"}
        </h1>

        <div className={css({ display: "flex", alignItems: "center", gap: "md" })}>
          <div className={css({ fontSize: "sm", color: "gray.600" })}>
            {layout === "split"
              ? "Showing: Video & Crops Mosaic (left), Non Focused (right)"
              : mode === "focused"
                ? "Showing: Crops Mosaic & Video"
                : "Showing: Non Focused Video"}
          </div>
          {headerRight}
        </div>
      </header>

      {/* Content */}
      {layout === "split" ? (
        // SPLIT MODE (3 streams at once)
        <div
          className={css({
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "row",
            gap: "md",
          })}
        >
          {/* LEFT: Focused (A + B) */}
          <section
            className={css({
              flex: 2,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              gap: "sm",
            })}
          >
            <h2 className={css({ fontSize: "lg", fontWeight: "700" })}>Focused</h2>
            <div className={css({ flex: 1, minHeight: 0 })}>
              <PatchedStreams
                key={`left-${connected ? "on" : "off"}`}
                topicGroups={topicGroups}
                defaultTopics={defaultOpenAll}
                allowedTopics={["Video", "Crops Mosaic"]}
                topicSortingFunction={sortAB}
              />
            </div>
          </section>

          {/* Divider */}
          <div className={css({ width: "2px", backgroundColor: "gray.300" })} />

          {/* RIGHT: Non Focused (C) */}
          <section
            className={css({
              flex: 1,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              gap: "sm",
            })}
          >
            <h2 className={css({ fontSize: "lg", fontWeight: "700" })}>Non Focused</h2>
            <div className={css({ flex: 1, minHeight: 0 })}>
              <PatchedStreams
                key={`right-${connected ? "on" : "off"}`}
                topicGroups={topicGroups}
                allowedTopics={["Non Focused Video"]}
              />
            </div>
          </section>
        </div>
      ) : (
        <>
          <div className={css({ flex: 1, minHeight: 0 })}>
            <Streams
              key={`${mode}-${connected ? "on" : "off"}`}
              topicGroups={topicGroups}
              defaultTopics={defaultTopics}
            />
          </div>

          <footer>
            <button
              onClick={() => setMode(m => (m === "focused" ? "nonFocused" : "focused"))}
              className={css({
                width: "100%",
                paddingY: "md",
                borderRadius: "lg",
                borderWidth: "1px",
                borderColor: "gray.300",
                backgroundColor: "transparent",
                _hover: { backgroundColor: "gray.100" },
                fontWeight: "600",
              })}
            >
              {mode === "focused" ? "Switch to Non-Focused" : "Switch to Focused"}
            </button>
          </footer>
        </>
      )}
    </main>
  );
}
