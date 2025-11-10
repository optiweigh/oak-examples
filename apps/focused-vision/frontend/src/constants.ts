// src/constants.ts

// 1) low-res with face detections overlayed
export const LOW_RES_TOPIC_GROUPS = {
  "640x640 RGB": "A",
  "NN detections": "A",
};

// what is visible by default on that panel
export const LOW_RES_DEFAULT_TOPICS = ["640x640 RGB", "NN detections"];

// we allow both to be turned on/off, but panel will only show these
export const LOW_RES_ALLOWED_TOPICS = ["640x640 RGB", "NN detections"];

// 2) non-focused head crops
export const NON_FOCUS_HEAD_CROPS_TOPIC_GROUPS = {
  "non_focus_head_crops": "A",
};
export const NON_FOCUS_HEAD_CROPS_DEFAULT = ["Non-Focus Head Crops"];
export const NON_FOCUS_HEAD_CROPS_ALLOWED = ["Non-Focus Head Crops"];

// 3) focused vision head crops
export const FOCUSED_VISION_HEAD_CROPS_TOPIC_GROUPS = {
  "focused_vision_head_crops": "A",
};
export const FOCUSED_VISION_HEAD_CROPS_DEFAULT = ["Focused Vision Head Crops"];
export const FOCUSED_VISION_HEAD_CROPS_ALLOWED = ["Focused Vision Head Crops"];