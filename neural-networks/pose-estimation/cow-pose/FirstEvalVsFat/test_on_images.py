"""
Test Cattle Segmentation on Saved Images

Runs inference on your training images to verify the model works.
Uses OpenCV DNN to run the ONNX model directly (no OAK camera needed).

Usage:
    python test_on_images.py                    # Test on images in current folder
    python test_on_images.py path/to/images     # Test on images in specified folder
"""
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.special import expit as sigmoid
import glob

# Paths - can be overridden by command line argument
if len(sys.argv) > 1:
    IMAGE_DIR = Path(sys.argv[1])
else:
    IMAGE_DIR = Path(__file__).parent
    
ONNX_PATH = Path(__file__).parent / "best.onnx"  # Model always in script folder

# Detection thresholds
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.4


def preprocess_image(img, input_size=(640, 640)):
    """Preprocess image for YOLOv8."""
    h, w = img.shape[:2]
    
    # Resize maintaining aspect ratio with letterboxing
    scale = min(input_size[0] / w, input_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create letterboxed image
    canvas = np.full((input_size[1], input_size[0], 3), 114, dtype=np.uint8)
    pad_x = (input_size[0] - new_w) // 2
    pad_y = (input_size[1] - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    
    # Convert to blob
    blob = cv2.dnn.blobFromImage(canvas, 1/255.0, input_size, swapRB=True, crop=False)
    
    return blob, canvas, (scale, pad_x, pad_y)


def decode_yolov8_seg(output0, output1, img_shape, preprocess_info, conf_thresh=0.25, iou_thresh=0.4):
    """
    Decode YOLOv8-seg outputs.
    """
    # Handle different output formats
    if output0.ndim == 3:
        predictions = output0[0].T  # (8400, 37)
        proto_masks = output1[0]    # (32, 160, 160)
    else:
        predictions = output0.T
        proto_masks = output1
    
    img_h, img_w = img_shape[:2]
    scale, pad_x, pad_y = preprocess_info
    
    # Parse predictions
    boxes = predictions[:, :4]           # x_center, y_center, w, h
    confidences = predictions[:, 4]      # confidence
    mask_coeffs = predictions[:, 5:37]   # 32 mask coefficients
    
    print(f"  Max confidence: {confidences.max():.4f}")
    print(f"  Confidences > {conf_thresh}: {(confidences > conf_thresh).sum()}")
    
    # Filter by confidence
    high_conf_mask = confidences > conf_thresh
    if not np.any(high_conf_mask):
        return []
    
    boxes = boxes[high_conf_mask]
    confidences = confidences[high_conf_mask]
    mask_coeffs = mask_coeffs[high_conf_mask]
    
    # Take top candidates
    top_k = min(50, len(confidences))
    top_indices = np.argsort(confidences)[-top_k:][::-1]
    boxes = boxes[top_indices]
    confidences = confidences[top_indices]
    mask_coeffs = mask_coeffs[top_indices]
    
    # Convert from xywh to xyxy (in 640x640 space)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), 
        confidences.tolist(), 
        conf_thresh, 
        iou_thresh
    )
    
    if len(indices) == 0:
        return []
    
    results = []
    for idx in indices[:5]:  # Max 5 detections
        if isinstance(idx, (list, tuple, np.ndarray)):
            idx = idx[0]
        
        bbox_640 = boxes_xyxy[idx].copy()
        conf = float(confidences[idx])
        
        # Scale bbox back to original image coordinates
        # First remove padding, then scale
        bbox = bbox_640.copy()
        bbox[[0, 2]] = (bbox[[0, 2]] - pad_x) / scale
        bbox[[1, 3]] = (bbox[[1, 3]] - pad_y) / scale
        bbox = np.clip(bbox, 0, [img_w, img_h, img_w, img_h]).astype(int)
        
        # Generate mask
        coeffs = mask_coeffs[idx]
        mask_proto = np.tensordot(coeffs, proto_masks, axes=([0], [0]))
        mask_proto = sigmoid(mask_proto)
        
        # Resize mask to 640x640 first
        mask_640 = cv2.resize(mask_proto.astype(np.float32), (640, 640))
        
        # Crop out the padding and resize to original
        mask_cropped = mask_640[pad_y:640-pad_y, pad_x:640-pad_x]
        if mask_cropped.size > 0:
            mask_resized = cv2.resize(mask_cropped, (img_w, img_h))
        else:
            mask_resized = cv2.resize(mask_640, (img_w, img_h))
        
        binary_mask = (mask_resized > 0.5).astype(np.uint8)
        
        results.append((bbox, conf, binary_mask))
    
    return results


def draw_results(img, results):
    """Draw detection results."""
    overlay = img.copy()
    
    for bbox, conf, mask in results:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        
        # Mask overlay
        mask_color = np.zeros_like(img)
        mask_color[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.4, 0)
        
        # Contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Box and label
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"Cattle {conf:.2f}"
        cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return overlay


def main():
    # Check for ONNX file
    if not ONNX_PATH.exists():
        print(f"ERROR: ONNX file not found at {ONNX_PATH}")
        print("Please copy best.onnx from your Colab training to this folder.")
        print("\nAlternatively, I'll try to use ultralytics to run inference...")
        
        # Try ultralytics
        try:
            from ultralytics import YOLO
            
            # Look for a .pt file
            pt_files = list(IMAGE_DIR.glob("*.pt"))
            if pt_files:
                model_path = pt_files[0]
            else:
                print("No .pt file found either. Please provide best.onnx or best.pt")
                return
                
            print(f"Using ultralytics with {model_path}")
            model = YOLO(str(model_path))
            
            # Get all images
            images = list(IMAGE_DIR.glob("*.jpg"))
            print(f"Found {len(images)} images")
            
            for img_path in images[:5]:  # Test first 5
                print(f"\nProcessing: {img_path.name}")
                results = model(str(img_path), conf=CONF_THRESHOLD)
                
                # Show results
                for r in results:
                    img = r.plot()
                    cv2.imshow("Result", img)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        return
                        
            cv2.destroyAllWindows()
            return
            
        except ImportError:
            print("ultralytics not installed. Please provide best.onnx file.")
            return
    
    # Load ONNX model with OpenCV DNN
    print(f"Loading ONNX model: {ONNX_PATH}")
    net = cv2.dnn.readNetFromONNX(str(ONNX_PATH))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Get all images
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images")
    
    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Preprocess
        blob, canvas, preprocess_info = preprocess_image(img)
        
        # Run inference
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        
        print(f"  Output shapes: {[o.shape for o in outputs]}")
        
        # Decode
        # YOLOv8-seg typically has 2 outputs
        if len(outputs) == 2:
            # Figure out which is which by shape
            if outputs[0].shape[-1] == 8400:
                output0, output1 = outputs[0], outputs[1]
            else:
                output0, output1 = outputs[1], outputs[0]
        else:
            print(f"  Unexpected number of outputs: {len(outputs)}")
            continue
        
        results = decode_yolov8_seg(
            output0, output1, img.shape, preprocess_info,
            conf_thresh=CONF_THRESHOLD, iou_thresh=IOU_THRESHOLD
        )
        
        print(f"  Detections: {len(results)}")
        
        # Draw and show
        display = draw_results(img, results)
        
        # Resize for display
        h, w = display.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            display = cv2.resize(display, (int(w*scale), int(h*scale)))
        
        cv2.imshow(f"Result - {img_path.name}", display)
        print("  Press any key for next, 'q' to quit, 's' to save")
        
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_path = img_path.parent / f"result_{img_path.name}"
            cv2.imwrite(str(out_path), display)
            print(f"  Saved: {out_path}")
    
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
