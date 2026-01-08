from typing import List, Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import depthai as dai
from depthai_nodes import Keypoints, SECONDARY_COLOR
from depthai_nodes.utils import AnnotationHelper

# Directory to save snapshots
SNAPSHOT_DIR = Path("snapshots")

# COCO keypoint indices
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12

# Threshold for side-on detection (ratio of horizontal to vertical spread)
SIDE_ON_THRESHOLD = 0.3  # Adjust this value to tune sensitivity

# Blur detection threshold (higher = sharper image required)
BLUR_THRESHOLD = 100.0  # Adjust based on your needs


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values = sharper image, lower values = more blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def is_side_on(xs: List[float], ys: List[float], confidences: List[float], conf_threshold: float) -> bool:
    """
    Detect if a person is standing side-on by analyzing shoulder and hip positions.
    When side-on, shoulders and hips will have minimal horizontal spread.
    """
    # Check if we have enough keypoints with sufficient confidence
    required_keypoints = [KEYPOINT_LEFT_SHOULDER, KEYPOINT_RIGHT_SHOULDER, 
                         KEYPOINT_LEFT_HIP, KEYPOINT_RIGHT_HIP]
    
    if len(xs) <= max(required_keypoints):
        return False
    
    for kp in required_keypoints:
        if confidences[kp] < conf_threshold:
            return False
    
    # Calculate horizontal distance between shoulders and hips
    shoulder_h_dist = abs(xs[KEYPOINT_LEFT_SHOULDER] - xs[KEYPOINT_RIGHT_SHOULDER])
    hip_h_dist = abs(xs[KEYPOINT_LEFT_HIP] - xs[KEYPOINT_RIGHT_HIP])
    
    # Calculate vertical distance (body height approximation)
    shoulder_mid_y = (ys[KEYPOINT_LEFT_SHOULDER] + ys[KEYPOINT_RIGHT_SHOULDER]) / 2
    hip_mid_y = (ys[KEYPOINT_LEFT_HIP] + ys[KEYPOINT_RIGHT_HIP]) / 2
    body_height = abs(hip_mid_y - shoulder_mid_y)
    
    if body_height < 0.01:  # Avoid division by zero
        return False
    
    # When side-on, horizontal spread will be small relative to body height
    shoulder_ratio = shoulder_h_dist / body_height
    hip_ratio = hip_h_dist / body_height
    
    # Person is side-on if both shoulders and hips have minimal horizontal spread
    return shoulder_ratio < SIDE_ON_THRESHOLD and hip_ratio < SIDE_ON_THRESHOLD


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.out_pose_annotations = self.createOutput()
        self.video_input = self.createInput()  # Separate input for video frames
        self.connection_pairs = [[]]
        self.valid_labels = [0]
        self.padding = 0.1
        self.keypoint_conf_threshold = 0.5
        self.save_snapshots = False
        self.snapshot_cooldown = 2.0  # Minimum seconds between snapshots
        self.last_snapshot_time = 0.0
        self.blur_threshold = BLUR_THRESHOLD

    def build(
        self,
        gather_data_msg: dai.Node.Output,
        connection_pairs: List[List[int]],
        valid_labels: List[int],
        padding: Optional[float] = None,
        keypoint_conf_threshold: Optional[float] = None,
        video_frame: Optional[dai.Node.Output] = None,
        snapshot_cooldown: Optional[float] = None,
        blur_threshold: Optional[float] = None,
    ) -> "AnnotationNode":
        self.connection_pairs = connection_pairs
        self.valid_labels = valid_labels
        if padding:
            self.padding = padding
        if keypoint_conf_threshold:
            self.keypoint_conf_threshold = keypoint_conf_threshold
        if snapshot_cooldown:
            self.snapshot_cooldown = snapshot_cooldown
        if blur_threshold:
            self.blur_threshold = blur_threshold
        
        # Enable snapshot saving if video frame is provided
        if video_frame:
            self.save_snapshots = True
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            video_frame.link(self.video_input)
            self.video_input.setBlocking(False)
            self.video_input.setMaxSize(1)
        
        self.link_args(gather_data_msg)
        return self

    def process(self, gather_data_msg: dai.Buffer) -> None:
        img_detections_msg: dai.ImgDetections = gather_data_msg.reference_data
        assert isinstance(img_detections_msg, dai.ImgDetections)

        keypoints_msg_list: List[Keypoints] = gather_data_msg.gathered
        assert isinstance(keypoints_msg_list, list)
        assert all(isinstance(msg, Keypoints) for msg in keypoints_msg_list)
        
        # Get video frame for snapshot if available
        frame_for_snapshot = None
        if self.save_snapshots:
            video_frame = self.video_input.tryGet()
            if video_frame is not None:
                frame_for_snapshot = video_frame.getCvFrame()

        annotations = AnnotationHelper()

        for img_detection_msg, keypoints_msg in zip(
            img_detections_msg.detections, keypoints_msg_list
        ):
            xmin, ymin, xmax, ymax = (
                img_detection_msg.xmin,
                img_detection_msg.ymin,
                img_detection_msg.xmax,
                img_detection_msg.ymax,
            )

            slope_x = (xmax + self.padding) - (xmin - self.padding)
            slope_y = (ymax + self.padding) - (ymin - self.padding)
            xs = []
            ys = []
            confidences = []
            for keypoint_msg in keypoints_msg.keypoints:
                x = min(
                    max(xmin - self.padding + slope_x * keypoint_msg.x, 0.0),
                    1.0,
                )
                y = min(
                    max(ymin - self.padding + slope_y * keypoint_msg.y, 0.0),
                    1.0,
                )
                xs.append(x)
                ys.append(y)
                confidences.append(keypoint_msg.confidence)

            # Check if person is standing side-on
            if is_side_on(xs, ys, confidences, self.keypoint_conf_threshold):
                current_time = datetime.now()
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] SIDE-ON DETECTED: Person is standing side-on to the camera")
                
                # Save snapshot with cooldown to avoid too many images
                current_time_sec = current_time.timestamp()
                if frame_for_snapshot is not None and (current_time_sec - self.last_snapshot_time) >= self.snapshot_cooldown:
                    # Check image sharpness before saving
                    blur_score = calculate_blur_score(frame_for_snapshot)
                    if blur_score >= self.blur_threshold:
                        filename = SNAPSHOT_DIR / f"sideon_{current_time.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                        cv2.imwrite(str(filename), frame_for_snapshot)
                        print(f"[{timestamp}] SNAPSHOT SAVED: {filename} (sharpness: {blur_score:.1f})")
                        self.last_snapshot_time = current_time_sec
                    else:
                        print(f"[{timestamp}] SNAPSHOT SKIPPED: Image too blurry (sharpness: {blur_score:.1f}, threshold: {self.blur_threshold})")
                
                # Add visual annotation near the detected person's center
                text_x = (xmin + xmax) / 2
                text_y = (ymin + ymax) / 2 - 0.05  # Slightly above center
                annotations.draw_text(
                    text="SIDE-ON",
                    position=(text_x, text_y),
                    color=SECONDARY_COLOR,
                    size=24,
                )

            for connection in self.connection_pairs:
                pt1_idx, pt2_idx = connection
                if (
                    confidences[pt1_idx] < self.keypoint_conf_threshold
                    or confidences[pt2_idx] < self.keypoint_conf_threshold
                ):
                    continue
                if pt1_idx < len(xs) and pt2_idx < len(xs):
                    x1, y1 = xs[pt1_idx], ys[pt1_idx]
                    x2, y2 = xs[pt2_idx], ys[pt2_idx]

                    annotations.draw_line([x1, y1], [x2, y2], thickness=1)
                    annotations.draw_circle(center=[x1, y1], radius=0.005)
                    annotations.draw_circle(center=[x2, y2], radius=0.005)

        img_annotations_msg = annotations.build(
            timestamp=img_detections_msg.getTimestamp(),
            sequence_num=img_detections_msg.getSequenceNum(),
        )

        self.out_pose_annotations.send(img_annotations_msg)
