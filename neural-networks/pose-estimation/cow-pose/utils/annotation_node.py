"""
Annotation Node for Cow Pose Estimation

This node:
1. Receives detection + pose data
2. Draws skeleton visualization
3. Saves snapshots when a cow is detected with sufficient confidence and image clarity

KEY CONCEPTS:

1. HostNode:
   - A node that runs on your computer (host), not on the OAK camera
   - Used for complex processing that needs Python libraries
   - Receives data from camera, processes it, sends results back

2. Keypoints:
   - Specific body part locations (head, legs, tail, etc.)
   - Each keypoint has x, y coordinates and a confidence score
   - Connected by "skeleton edges" to form a stick figure

3. Blur Detection (Laplacian Variance):
   - Sharp images have lots of edges (high variance)
   - Blurry images have few edges (low variance)
   - We skip saving blurry images
"""

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
# Typical values: 50-200 depending on camera and scene
BLUR_THRESHOLD = 100.0


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    
    The Laplacian operator detects edges in an image.
    Sharp images have many strong edges = high variance.
    Blurry images have weak edges = low variance.
    
    Args:
        image: BGR image as numpy array
        
    Returns:
        Variance of the Laplacian (higher = sharper)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


class AnnotationNode(dai.node.HostNode):
    """
    Custom node that processes detection + pose results.
    
    Inherits from dai.node.HostNode, which means:
    - It runs on your computer, not the camera
    - It can use any Python library (OpenCV, numpy, etc.)
    - It processes data in the `process()` method
    """
    
    def __init__(self) -> None:
        super().__init__()
        # Create outputs that other nodes can connect to
        self.video_input = self.createInput()  # For receiving video frames
        self.out_detections = self.createOutput()  # Sends detection results
        self.out_pose_annotations = self.createOutput(  # Sends pose visualization
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        
        # Configuration
        self.connection_pairs = [[]]  # Which keypoints to connect with lines
        self.padding = 0.1
        self.save_snapshots = False
        self.snapshot_cooldown = 2.0  # Seconds between snapshots
        self.last_snapshot_time = 0.0
        self.blur_threshold = BLUR_THRESHOLD
        self.confidence_threshold = 0.5  # Minimum detection confidence for snapshots

    def build(
        self,
        input_detections: dai.Node.Output,
        connection_pairs: List[List[int]],
        padding: float,
        video_frame: Optional[dai.Node.Output] = None,
        snapshot_cooldown: Optional[float] = None,
        blur_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
    ) -> "AnnotationNode":
        """
        Configure the node and connect inputs.
        
        Args:
            input_detections: Where detection+pose data comes from
            connection_pairs: List of [keypoint1, keypoint2] pairs for skeleton
            padding: Extra padding around detections
            video_frame: High-res video for snapshots (optional)
            snapshot_cooldown: Minimum seconds between snapshots
            blur_threshold: Minimum sharpness to save a snapshot
            confidence_threshold: Minimum detection confidence to save a snapshot
        """
        self.connection_pairs = connection_pairs
        self.padding = padding
        
        if snapshot_cooldown is not None:
            self.snapshot_cooldown = snapshot_cooldown
        if blur_threshold is not None:
            self.blur_threshold = blur_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        # Enable snapshot saving if video frame is provided
        if video_frame:
            self.save_snapshots = True
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            video_frame.link(self.video_input)
            self.video_input.setBlocking(False)
            self.video_input.setMaxSize(1)
        
        # Link the main input (detection + pose data)
        self.link_args(input_detections)
        return self

    def process(self, gathered_data: dai.Buffer) -> None:
        """
        Process each frame of detection + pose data.
        
        This method is called automatically for each new frame.
        
        Args:
            gathered_data: Contains detections and their corresponding pose estimates
        """
        assert isinstance(gathered_data, GatheredData)

        # Get the detection results
        detections_message: ImgDetectionsExtended = gathered_data.reference_data
        detections_list: List[ImgDetectionExtended] = detections_message.detections

        # Create helper for drawing annotations
        annotation_helper = AnnotationHelper()
        padding = self.padding
        
        # Get video frame for snapshot if available
        frame_for_snapshot = None
        if self.save_snapshots:
            video_frame = self.video_input.tryGet()
            if video_frame is not None:
                frame_for_snapshot = video_frame.getCvFrame()

        # Process each detected cow
        for ix, detection in enumerate(detections_list):
            detection.label_name = "Cow"  # Set the label name for display

            # Get the pose keypoints for this detection
            keypoints_message: Keypoints = gathered_data.gathered[ix]
            
            # Get bounding box coordinates (normalized 0-1)
            xmin, ymin, xmax, ymax = detection.rotated_rect.getOuterRect()

            # Calculate scaling factors for keypoint positions
            slope_x = (xmax + padding) - (xmin - padding)
            slope_y = (ymax + padding) - (ymin - padding)
            
            # Convert keypoints from crop coordinates to full image coordinates
            xs = []
            ys = []
            for kp in keypoints_message.keypoints:
                x = min(max(xmin - padding + slope_x * kp.x, 0.0), 1.0)
                y = min(max(ymin - padding + slope_y * kp.y, 0.0), 1.0)
                xs.append(x)
                ys.append(y)

            # Log detection and potentially save snapshot
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            confidence = detection.confidence
            print(f"[{timestamp}] COW DETECTED (confidence: {confidence*100:.1f}%)")
            
            # Save snapshot if conditions are met
            current_time_sec = current_time.timestamp()
            if confidence < self.confidence_threshold:
                print(f"[{timestamp}] SNAPSHOT SKIPPED: Confidence too low ({confidence*100:.1f}% < {self.confidence_threshold*100:.0f}%)")
            elif frame_for_snapshot is not None and (current_time_sec - self.last_snapshot_time) >= self.snapshot_cooldown:
                blur_score = calculate_blur_score(frame_for_snapshot)
                if blur_score >= self.blur_threshold:
                    filename = SNAPSHOT_DIR / f"cow_{current_time.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    cv2.imwrite(str(filename), frame_for_snapshot)
                    print(f"[{timestamp}] SNAPSHOT SAVED: {filename} (sharpness: {blur_score:.1f})")
                    self.last_snapshot_time = current_time_sec
                else:
                    print(f"[{timestamp}] SNAPSHOT SKIPPED: Image too blurry (sharpness: {blur_score:.1f}, threshold: {self.blur_threshold})")

            # Draw the skeleton (connect keypoints with lines)
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

            # Draw the keypoint dots
            kpts_to_draw = [(xs[i], ys[i]) for i in kpts_to_draw]
            annotation_helper.draw_points(
                points=kpts_to_draw,
                color=PRIMARY_COLOR,
                thickness=2.0,
            )

        # Build and send the annotation message
        annotations = annotation_helper.build(
            timestamp=detections_message.getTimestamp(),
            sequence_num=detections_message.getSequenceNum(),
        )

        self.out_detections.send(detections_message)
        self.out_pose_annotations.send(annotations)
