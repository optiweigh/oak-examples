from typing import List, Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import depthai as dai
from depthai_nodes import (
    ImgDetectionsExtended,
    ImgDetectionExtended,
    Keypoints,
    GatheredData,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
)
from depthai_nodes.utils import AnnotationHelper

# Directory to save snapshots
SNAPSHOT_DIR = Path("snapshots")

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


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_keypoints = self.createInput()
        self.video_input = self.createInput()  # Separate input for video frames
        self.out_detections = self.createOutput()
        self.out_pose_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.connection_pairs = [[]]
        self.padding = 0.1
        self.save_snapshots = False
        self.snapshot_cooldown = 2.0  # Minimum seconds between snapshots
        self.last_snapshot_time = 0.0
        self.blur_threshold = BLUR_THRESHOLD

    def build(
        self,
        input_detections: dai.Node.Output,
        connection_pairs: List[List[int]],
        padding: float,
        video_frame: Optional[dai.Node.Output] = None,
        snapshot_cooldown: Optional[float] = None,
        blur_threshold: Optional[float] = None,
    ) -> "AnnotationNode":
        self.connection_pairs = connection_pairs
        self.padding = padding
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
        
        self.link_args(input_detections)
        return self

    def process(self, gathered_data: dai.Buffer) -> None:
        assert isinstance(gathered_data, GatheredData)

        detections_message: ImgDetectionsExtended = gathered_data.reference_data

        detections_list: List[ImgDetectionExtended] = detections_message.detections

        annotation_helper = AnnotationHelper()

        padding = self.padding
        
        # Get video frame for snapshot if available
        frame_for_snapshot = None
        if self.save_snapshots:
            video_frame = self.video_input.tryGet()
            if video_frame is not None:
                frame_for_snapshot = video_frame.getCvFrame()

        for ix, detection in enumerate(detections_list):
            detection.label_name = (
                "Animal"  # Because dai.ImgDetection does not have label_name
            )

            keypoints_message: Keypoints = gathered_data.gathered[ix]
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()

            slope_x = (xmax + padding) - (xmin - padding)
            slope_y = (ymax + padding) - (ymin - padding)
            xs = []
            ys = []
            for kp in keypoints_message.keypoints:
                x = min(max(xmin - padding + slope_x * kp.x, 0.0), 1.0)
                y = min(max(ymin - padding + slope_y * kp.y, 0.0), 1.0)
                xs.append(x)
                ys.append(y)

            # Cow detected - take snapshot if image is sharp enough and confidence is high
            # (The model only detects cows when side-on, so no need for side-on heuristics)
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            confidence = detection.confidence
            print(f"[{timestamp}] COW DETECTED (confidence: {confidence*100:.1f}%)")
            
            # Save snapshot with cooldown, confidence check, and blur check
            current_time_sec = current_time.timestamp()
            if confidence < 0.7:
                print(f"[{timestamp}] SNAPSHOT SKIPPED: Confidence too low ({confidence*100:.1f}% < 70%)")
            elif frame_for_snapshot is not None and (current_time_sec - self.last_snapshot_time) >= self.snapshot_cooldown:
                blur_score = calculate_blur_score(frame_for_snapshot)
                if blur_score >= self.blur_threshold:
                    filename = SNAPSHOT_DIR / f"cow_{current_time.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(str(filename), frame_for_snapshot)
                    print(f"[{timestamp}] SNAPSHOT SAVED: {filename} (sharpness: {blur_score:.1f})")
                    self.last_snapshot_time = current_time_sec
                else:
                    print(f"[{timestamp}] SNAPSHOT SKIPPED: Image too blurry (sharpness: {blur_score:.1f}, threshold: {self.blur_threshold})")

            kpts_to_draw = set()

            for connection in self.connection_pairs:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(xs) and pt2_idx < len(xs):
                    x1, y1 = xs[pt1_idx], ys[pt1_idx]
                    x2, y2 = xs[pt2_idx], ys[pt2_idx]
                    kpts_to_draw.add(pt1_idx)
                    kpts_to_draw.add(pt2_idx)
                    annotation_helper.draw_line(
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=SECONDARY_COLOR,
                        thickness=1.0,
                    )

            kpts_to_draw = [(xs[i], ys[i]) for i in kpts_to_draw]
            annotation_helper.draw_points(
                points=kpts_to_draw,
                color=PRIMARY_COLOR,
                thickness=2.0,
            )

        annotations = annotation_helper.build(
            timestamp=detections_message.getTimestamp(),
            sequence_num=detections_message.getSequenceNum(),
        )

        self.out_detections.send(detections_message)
        self.out_pose_annotations.send(annotations)
