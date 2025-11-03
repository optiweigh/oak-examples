export const topicGroups = {
  Video: "A",
  "Full Frame eyes detection": "A",
  "Crops Mosaic": "B",
  "Detections Stage 2 Crops": "B",
  "Eyes Mosaic": "C",
  "Eyes Mosaic Non Focused": "D",
  "Detections NN Non Focused": "E",
  "Detections Non Focused Remapped": "F",
};

export const VIDEO_TOPICS_FOCUSED = ["Video", "Full Frame eyes detection"];
export const VIDEO_TOPICS_NON_FOCUSED = ["Video", "Detections Non Focused Remapped"];
export const FACE_MOSAIC_TOPICS = ["Crops Mosaic", "Detections Stage 2 Crops"];
export const EYES_MOSAIC_FOCUSED_TOPICS = ["Eyes Mosaic"];
export const EYES_MOSAIC_NON_FOCUSED_TOPICS = ["Eyes Mosaic Non Focused"];
export const NN_INPUT_FACE_TOPICS = ["NN input Face Detection", "Detections Stage 1"];
export const NN_INPUT_EYE_TOPICS = ["NN input Eye Detection", "Detections NN Non Focused"];

export const defaultOpenAllFocused = ["Video", "Detections Stage 1", "Crops Mosaic", "Detections Stage 2 Crops", "Eyes Mosaic", "NN input Face Detection", "Full Frame eyes detection"];
export const defaultOpenAllNonFocused = ["Video", "Detections NN Non Focused", "Eyes Mosaic Non Focused", "NN input Eye Detection", "Detections Non Focused Remapped"];
