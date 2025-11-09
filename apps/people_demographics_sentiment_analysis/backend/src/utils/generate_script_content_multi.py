from typing import Optional, List


def generate_script_multi(
    emotions_w: int, emotions_h: int,
    age_gender_w: int, age_gender_h: int,
    reid_w: int, reid_h: int,
    *,
    resize_mode: str = "LETTERBOX",
    padding: float = 0,
    valid_labels: Optional[List[int]] = None,
) -> str:
    """
    The function generates the script content for the dai.Script node.

    It emits three crop/resize streams (emotions, age_gender, reid)
    from a single preview frame using the same resize policy and padding.
    """
    if resize_mode not in ("CENTER_CROP", "LETTERBOX", "NONE", "STRETCH"):
        raise ValueError("Unsupported resize mode")

    validate_label = (
        f"if detection.label not in {list(valid_labels)}: continue"
        if valid_labels is not None else ""
    )

    return f"""

def make_cfg(detection, out_w, out_h, pad):
    cfg = ImageManipConfig()
    rect = RotatedRect()
    rect.center.x = (detection.xmin + detection.xmax) * 0.5
    rect.center.y = (detection.ymin + detection.ymax) * 0.5
    rect.size.width  = (detection.xmax - detection.xmin) + (pad * 2.0)
    rect.size.height = (detection.ymax - detection.ymin) + (pad * 2.0)
    rect.angle = 0
    cfg.addCropRotatedRect(rect, True)
    cfg.setOutputSize(out_w, out_h, ImageManipConfig.ResizeMode.{resize_mode})
    return cfg

try:
    while True:
        frame = node.inputs['preview'].get()
        dets  = node.inputs['det_in'].get()

        for detection in dets.detections:
            {validate_label}

            cfg_emotions = make_cfg(detection, {emotions_w}, {emotions_h}, {padding})
            cfg_age_gender = make_cfg(detection, {age_gender_w}, {age_gender_h}, {padding})
            cfg_reid = make_cfg(detection, {reid_w}, {reid_h}, {padding})

            node.outputs['manip_cfg_emotions'].send(cfg_emotions)
            node.outputs['manip_img_emotions'].send(frame)

            node.outputs['manip_cfg_age_gender'].send(cfg_age_gender)
            node.outputs['manip_img_age_gender'].send(frame)

            node.outputs['manip_cfg_reid'].send(cfg_reid)
            node.outputs['manip_img_reid'].send(frame)

except Exception as e:
    node.warn(str(e))
"""
