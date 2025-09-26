// Use a handwriting-like font only for captions (add the Google Fonts link in index.html).
export const CAPTION_FONT = "'Patrick Hand', 'Caveat', 'Gloria Hallelujah', cursive";

export const topicGroups = {
  Video: "A",
  "Detections Stage 1": "A",
  "Crops Mosaic": "B",
  "Detections Stage 2 Crops": "B",
  "Eyes Mosaic": "C",
  "Eyes Mosaic Non Focused": "D",
  "Detections Non Focused": "E",
};

// default topics for each stream - only the specific topics for each mode
export const VIDEO_TOPICS_FOCUSED = ["Video", "Detections Stage 1"];
export const VIDEO_TOPICS_NON_FOCUSED = ["Video", "Detections Non Focused"];
export const FACE_MOSAIC_TOPICS = ["Crops Mosaic", "Detections Stage 2 Crops"];
export const EYES_MOSAIC_FOCUSED_TOPICS = ["Eyes Mosaic"];
export const EYES_MOSAIC_NON_FOCUSED_TOPICS = ["Eyes Mosaic Non Focused"];

// Create separate defaultOpenAll arrays for each mode
export const defaultOpenAllFocused = ["Video", "Detections Stage 1", "Crops Mosaic", "Detections Stage 2 Crops", "Eyes Mosaic"];
export const defaultOpenAllNonFocused = ["Video", "Detections Non Focused", "Eyes Mosaic Non Focused"];

// Separate arrays for video-only streams (no annotations)
export const defaultOpenAllVideoOnlyFocused = ["Video", "Crops Mosaic", "Detections Stage 2 Crops", "Eyes Mosaic"];
export const defaultOpenAllVideoOnlyNonFocused = ["Video", "Eyes Mosaic Non Focused"];

// Minimal arrays with only the specific topics we want
export const videoOnlyFocused = ["Video"];
export const videoOnlyNonFocused = ["Video"];

// all streams together (like the old defaultOpenAll)
export const defaultOpenAll = ["Video", "Crops Mosaic", "Eyes Mosaic"];