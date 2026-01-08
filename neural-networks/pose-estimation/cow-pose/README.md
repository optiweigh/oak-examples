# Cow Pose Estimation

This example detects **cows specifically** and estimates their pose (body keypoints).

## How It Works

### Two-Stage Pipeline

```
Camera Frame
     ↓
┌─────────────────────────────────┐
│  Stage 1: YOLOv6-nano (COCO)    │  ← Detects objects, filters for "cow"
│  - Trained on 80 COCO classes   │
│  - We filter to keep only cows  │
└─────────────────────────────────┘
     ↓ (cow bounding boxes)
┌─────────────────────────────────┐
│  Stage 2: SuperAnimal Landmarker│  ← Estimates pose on each detected cow
│  - Finds body keypoints         │
│  - Head, legs, tail, spine, etc │
└─────────────────────────────────┘
     ↓ (keypoints)
┌─────────────────────────────────┐
│  Annotation + Snapshot          │  ← Draws skeleton, saves clear photos
│  - Blur detection               │
│  - Confidence filtering         │
└─────────────────────────────────┘
```

### Key Concepts

- **COCO Dataset**: A famous ML dataset with 80 classes including animals
- **Detection Filtering**: We keep only cow detections, discarding people, cars, etc.
- **Pose Estimation**: Finding body part locations (keypoints) on detected animals
- **Blur Detection**: Laplacian variance to skip blurry frames

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run with camera
python main.py

# Run with video file
python main.py -media path/to/cow_video.mp4

# Open http://localhost:8082 to view the visualization
```

## Outputs

- **Console**: Logs each cow detection with confidence and snapshot status
- **Visualizer**: Shows video with pose skeleton overlay
- **Snapshots**: Saved to `snapshots/` folder when a clear cow is detected

## Configuration

Edit `main.py` to adjust:
- `TARGET_ANIMAL = "cow"` - Change to any COCO animal class (see table below)
- `PADDING = 0.1` - Extra padding around detections for pose estimation

Edit `utils/annotation_node.py` to adjust:
- `BLUR_THRESHOLD = 100.0` - Higher = require sharper images
- `snapshot_cooldown = 2.0` - Seconds between snapshots
- `confidence_threshold = 0.5` - Minimum detection confidence (50%)

## COCO Animal Classes

The model automatically finds the correct class ID for the animal name you specify.
Available animals in COCO:

| Animal     | Notes |
|------------|-------|
| bird       | General birds |
| cat        | Domestic cats |
| dog        | Domestic dogs |
| horse      | Horses |
| sheep      | Sheep |
| **cow**    | **Cattle (default)** |
| elephant   | Elephants |
| bear       | Bears |
| zebra      | Zebras |
| giraffe    | Giraffes |

To detect a different animal, simply change `TARGET_ANIMAL` in `main.py`:
```python
TARGET_ANIMAL = "sheep"  # or "horse", "dog", etc.
```
