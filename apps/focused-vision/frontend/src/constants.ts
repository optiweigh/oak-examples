// src/constants.ts

// 1) low-res with face detections overlayed
export const LOW_RES_TOPIC_GROUPS = {
  "low_res_image": "A",
  "face_detections": "A",
};

// what is visible by default on that panel
export const LOW_RES_DEFAULT_TOPICS = ["low_res_image", "face_detections"];

// we allow both to be turned on/off, but panel will only show these
export const LOW_RES_ALLOWED_TOPICS = ["low_res_image", "face_detections"];

// 2) non-focused head crops
export const NON_FOCUS_HEAD_CROPS_TOPIC_GROUPS = {
  "non_focus_head_crops": "A",
};
export const NON_FOCUS_HEAD_CROPS_DEFAULT = ["non_focus_head_crops"];
export const NON_FOCUS_HEAD_CROPS_ALLOWED = ["non_focus_head_crops"];

// 3) focused vision head crops
export const FOCUSED_VISION_HEAD_CROPS_TOPIC_GROUPS = {
  "focused_vision_head_crops": "A",
};
export const FOCUSED_VISION_HEAD_CROPS_DEFAULT = ["focused_vision_head_crops"];
export const FOCUSED_VISION_HEAD_CROPS_ALLOWED = ["focused_vision_head_crops"];