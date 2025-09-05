import { css } from "../styled-system/css/css.mjs";
import { Streams } from "@luxonis/depthai-viewer-common";

// keep each annotation on its stream
const topicGroups = {
  "Video": "A",
  "Face stage 1": "A",
  "Face Mosaic": "B",
  "Eyes (Crops)": "B",
  "Non Focused Video": "C",
  "Eyes (Non-Focused)": "C",
};

// open all three once (left column will do this)
const defaultOpenAll = ["Video", "Face Mosaic", "Non Focused Video"];

// order inside the left column (change to ["Face Mosaic","Video"] if you prefer)
const ORDER_AB = ["Video", "Face Mosaic"];
const sortAB = (a: { name: string }, b: { name: string }) => {
  const ai = ORDER_AB.indexOf(a.name);
  const bi = ORDER_AB.indexOf(b.name);
  return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
};

export default function App() {
  return (
    <main
      className={css({
        width: "screen",
        height: "screen",
        display: "flex",
        flexDirection: "row",
        gap: "md",
        padding: "md",
      })}
    >
      {/* LEFT: Focused (A + B) */}
      <section
        className={css({
          flex: 2,            // wider column for two panels
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          gap: "sm",
        })}
      >
        <h2 className={css({ fontSize: "xl", fontWeight: "700" })}>Focused</h2>
        <div className={css({ flex: 1, minHeight: 0 })}>
          <Streams
            topicGroups={topicGroups}
            defaultTopics={defaultOpenAll}              // open Video + Face Mosaic + Non Focused
            allowedTopics={["Video", "Face Mosaic"]}    // show only A+B on the left
            topicSortingFunction={sortAB}               // order A then B
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
        <h2 className={css({ fontSize: "xl", fontWeight: "700" })}>Non Focused</h2>
        <div className={css({ flex: 1, minHeight: 0 })}>
          <Streams
            topicGroups={topicGroups}
            allowedTopics={["Non Focused Video"]}       // show only C on the right
            // NOTE: no defaultTopics here -> avoids stopTopics() racing the left column
          />
        </div>
      </section>
    </main>
  );
}
