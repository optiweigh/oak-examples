// src/components/LuxonisStream.tsx
import React from "react";
import { Streams } from "@luxonis/depthai-viewer-common";

// the viewer component isn’t always nicely typed for custom props
const PatchedStreams = Streams as unknown as React.ComponentType<any>;

interface LuxonisStreamProps {
  title: string;
  defaultTopics: string[];
  allowedTopics: string[];
  // for your case: low_res_image uses detections, others don’t
  topicGroups?: Record<string, string>;
  hideToolbar?: boolean;
  connected?: boolean;
}

const LuxonisStream: React.FC<LuxonisStreamProps> = ({
  title,
  defaultTopics,
  allowedTopics,
  topicGroups = {},
}) => {
  return (
    <section className="stream-card">
      <h2 className="stream-title">{title}</h2>
      <div className="stream-body">
        <PatchedStreams
          topicGroups={topicGroups}
          defaultTopics={defaultTopics}
          allowedTopics={allowedTopics}
          hideToolbar
        />
      </div>
    </section>
  );
};

export default LuxonisStream;