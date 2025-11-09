// src/App.tsx
import LuxonisStream from "./components/LuxonisStream";
import {
  LOW_RES_TOPIC_GROUPS,
  LOW_RES_DEFAULT_TOPICS,
  LOW_RES_ALLOWED_TOPICS,
  NON_FOCUS_HEAD_CROPS_TOPIC_GROUPS,
  NON_FOCUS_HEAD_CROPS_DEFAULT,
  NON_FOCUS_HEAD_CROPS_ALLOWED,
  FOCUSED_VISION_HEAD_CROPS_TOPIC_GROUPS,
  FOCUSED_VISION_HEAD_CROPS_DEFAULT,
  FOCUSED_VISION_HEAD_CROPS_ALLOWED,
} from "./constants";

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
        <LuxonisStream
          title="Low-res image (with face detections)"
          topicGroups={LOW_RES_TOPIC_GROUPS}
          defaultTopics={LOW_RES_DEFAULT_TOPICS}
          allowedTopics={LOW_RES_ALLOWED_TOPICS}
        />

        {/* 2) non-focused head crops */}
        <LuxonisStream
          title="Non-focus head crops"
          topicGroups={NON_FOCUS_HEAD_CROPS_TOPIC_GROUPS}
          defaultTopics={NON_FOCUS_HEAD_CROPS_DEFAULT}
          allowedTopics={NON_FOCUS_HEAD_CROPS_ALLOWED}
        />

        {/* 3) focused vision head crops */}
        <LuxonisStream
          title="Focused vision head crops"
          topicGroups={FOCUSED_VISION_HEAD_CROPS_TOPIC_GROUPS}
          defaultTopics={FOCUSED_VISION_HEAD_CROPS_DEFAULT}
          allowedTopics={FOCUSED_VISION_HEAD_CROPS_ALLOWED}
        />
      </main>
    </div>
  );
};
