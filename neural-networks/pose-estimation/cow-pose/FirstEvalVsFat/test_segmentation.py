"""
Test Cattle Segmentation Model

Runs your custom-trained YOLOv8 segmentation model on the OAK-1-W.
This uses the raw NeuralNetwork node with host-side decoding.
"""
from pathlib import Path

import depthai as dai
import cv2
import numpy as np
from scipy.special import expit as sigmoid

# Path to your trained model
MODEL_PATH = Path(__file__).parent / "best.rvc2.tar.xz"

# Detection thresholds
CONF_THRESHOLD = 0.25  # Lower threshold since model trained on limited data
IOU_THRESHOLD = 0.4


def decode_yolov8_seg_fast(output0, output1, conf_thresh=0.25, iou_thresh=0.4, max_det=5):
    """
    Fast decode of YOLOv8-seg outputs.
    
    output0: (1, 37, 8400) - detections [4 bbox + 1 conf + 32 mask coeffs]
    output1: (1, 32, 160, 160) - mask prototypes
    
    Returns results in 640x640 space (model input space)
    """
    predictions = output0[0].T  # (8400, 37)
    proto_masks = output1[0]    # (32, 160, 160)
    
    # Parse predictions
    boxes = predictions[:, :4]           # x_center, y_center, w, h
    confidences = predictions[:, 4]      # confidence
    mask_coeffs = predictions[:, 5:37]   # 32 mask coefficients
    
    # Pre-filter by confidence (quick filter)
    high_conf_mask = confidences > conf_thresh
    if not np.any(high_conf_mask):
        return []
    
    boxes = boxes[high_conf_mask]
    confidences = confidences[high_conf_mask]
    mask_coeffs = mask_coeffs[high_conf_mask]
    
    # Sort by confidence and take top candidates
    top_k = min(100, len(confidences))
    top_indices = np.argsort(confidences)[-top_k:][::-1]
    boxes = boxes[top_indices]
    confidences = confidences[top_indices]
    mask_coeffs = mask_coeffs[top_indices]
    
    # Convert from xywh to xyxy (in 640x640 space)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), 
        confidences.tolist(), 
        conf_thresh, 
        iou_thresh
    )
    
    if len(indices) == 0:
        return []
    
    # Limit max detections
    indices = indices[:max_det]
    
    results = []
    for idx in indices:
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]
        
        bbox = boxes_xyxy[idx].astype(int)
        conf = float(confidences[idx])
        
        # Generate mask from prototype (in 160x160 space)
        coeffs = mask_coeffs[idx]  # (32,)
        mask_proto = np.tensordot(coeffs, proto_masks, axes=([0], [0]))  # (160, 160)
        
        # Apply sigmoid
        mask_proto = sigmoid(mask_proto)
        
        # Resize to 640x640
        mask_640 = cv2.resize(mask_proto.astype(np.float32), (640, 640))
        
        # Threshold to binary mask
        binary_mask = (mask_640 > 0.5).astype(np.uint8)
        
        results.append((bbox, conf, binary_mask))
    
    return results


def draw_results(img, results):
    """Draw detection results on 640x640 image."""
    overlay = img.copy()
    
    for item in results:
        if len(item) == 3:
            bbox, conf, mask = item
            color = (0, 255, 0)
            x1, y1, x2, y2 = bbox
            
            # Draw mask if available
            if mask is not None:
                mask_color = np.zeros_like(img)
                mask_color[mask > 0] = color
                overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.3, 0)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
        else:
            bbox, conf = item[:2]
            mask = None
            color = (0, 255, 0)
            x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"Cattle {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1-h-10), (x1+w+4, y1), color, -1)
        cv2.putText(overlay, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return overlay


def decode_boxes_only(output0, conf_thresh=0.25, iou_thresh=0.4, max_det=5):
    """Decode just bounding boxes without masks (faster for debugging)."""
    predictions = output0[0].T  # (8400, 37)
    
    boxes = predictions[:, :4]
    confidences = predictions[:, 4]
    
    # Filter
    high_conf_mask = confidences > conf_thresh
    if not np.any(high_conf_mask):
        return []
    
    boxes = boxes[high_conf_mask]
    confidences = confidences[high_conf_mask]
    
    # Top-k
    top_k = min(100, len(confidences))
    top_indices = np.argsort(confidences)[-top_k:][::-1]
    boxes = boxes[top_indices]
    confidences = confidences[top_indices]
    
    # xywh to xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), conf_thresh, iou_thresh)
    
    if len(indices) == 0:
        return []
    
    results = []
    for idx in indices[:max_det]:
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]
        bbox = boxes_xyxy[idx].astype(int)
        conf = float(confidences[idx])
        results.append((bbox, conf))
    
    return results


print(f"Loading model from: {MODEL_PATH}")
print("Connecting to camera...")

device = dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

# Check model expected input
nn_archive = dai.NNArchive(archivePath=str(MODEL_PATH))
config = nn_archive.getConfig()
inp_config = config.model.inputs[0]
print(f"Model expects: dtype={inp_config.dtype}, layout={inp_config.layout}")
print(f"Preprocessing: daiType={inp_config.preprocessing.daiType}")

# Use BGR (model expects BGR888p)
frame_type = dai.ImgFrame.Type.BGR888p

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # Model already loaded above
    input_size = nn_archive.getInputSize()
    print(f"Model input size: {input_size}")
    
    # Camera - request exactly 640x640 to match model input
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((640, 640), frame_type, fps=10)
    
    # ImageManip to ensure proper format
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(640, 640)
    manip.initialConfig.setFrameType(frame_type)
    manip.setMaxOutputFrameSize(640 * 640 * 3 + 10000)
    cam_out.link(manip.inputImage)

    # Run neural network
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setNNArchive(nn_archive)
    manip.out.link(nn.input)
    
    # Output queues - get the manipulated image that matches NN input
    manip_q = manip.out.createOutputQueue()
    nn_q = nn.out.createOutputQueue()

    print("Pipeline created successfully!")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("=" * 50)
    
    pipeline.start()
    
    frame_count = 0
    display = None
    debug_printed = False

    while pipeline.isRunning():
        # Get the preprocessed frame (same as what NN sees)
        manip_frame = manip_q.tryGet()
        if manip_frame is not None:
            # Get frame - already BGR
            img_bgr = manip_frame.getCvFrame()
            display = img_bgr.copy()
            frame_count += 1
            
            # Get NN output
            nn_out = nn_q.tryGet()
            if nn_out is not None:
                # Get tensors
                output0 = nn_out.getTensor("output0")
                output1 = nn_out.getTensor("output1")
                
                # Debug: print tensor info once
                if not debug_printed:
                    debug_printed = True
                    print(f"\n=== Tensor Debug ===")
                    print(f"output0 shape: {output0.shape}, dtype: {output0.dtype}")
                    print(f"output0 min/max: {float(np.min(output0)):.4f} / {float(np.max(output0)):.4f}")
                    print(f"output1 shape: {output1.shape}, dtype: {output1.dtype}")
                    print(f"output1 min/max: {float(np.min(output1)):.4f} / {float(np.max(output1)):.4f}")
                    print("=" * 40)
                
                # Skip mask processing for now - just show boxes
                results = decode_boxes_only(output0, CONF_THRESHOLD, IOU_THRESHOLD)
                
                # Draw results on BGR image
                display = draw_results(img_bgr, results)
                
                # Show info
                info = f"Detections: {len(results)} | Frame: {frame_count}"
                cv2.putText(display, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow("Cattle Segmentation", display)
        
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("s") and display is not None:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cattle_seg_{ts}.jpg"
            cv2.imwrite(filename, display)
            print(f"Saved: {filename}")

cv2.destroyAllWindows()
print("Done!")
