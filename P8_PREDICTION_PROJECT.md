# P8 Fat Prediction from Backline Images

## R&D Learning Project Outline

**Created**: December 24, 2025  
**Purpose**: Learn deep learning by building a practical cattle carcass score prediction system

---

## Project Goal

Predict P8 fat measurement (in mm) from images of a cow's backline captured from an elevated position at a cattle feeder.

```
┌─────────────────┐                    ┌─────────────────┐
│  Backline Image │  ──── MODEL ────>  │  P8 = 14.2mm    │
│  (from feeder)  │                    │                 │
└─────────────────┘                    └─────────────────┘
```

---

## Data Sources

| Data | Source | Notes |
|------|--------|-------|
| **Images** | OAK camera at feeder | Elevated angle, consistent framing |
| **Animal ID** | RFID at feeder | Links image to specific animal |
| **P8 Score** | Kill data | Actual P8 measurement in mm |
| **Weight** | Scales / records | Live weight at time of image (kg) |
| **Age Range** | Records | e.g., 12-18 months, 18-24 months |
| **Breed Type** | Records | e.g., Angus, Hereford, Crossbred |

### Master CSV Format

```csv
filename,animal_id,p8_mm,weight_kg,age_range,breed
cow_001.jpg,RFID123456,14.2,580,18-24m,angus
cow_002.jpg,RFID789012,8.5,520,12-18m,hereford
cow_003.jpg,RFID345678,22.0,640,24-30m,crossbred
```

### Using Additional Features

These extra data points can improve predictions:

**Option A: Multi-input Model**
```
┌─────────────────┐
│  Masked Image   │───┐
└─────────────────┘   │    ┌──────────────┐
                      ├──> │   COMBINED   │──> P8 Prediction
┌─────────────────┐   │    │    MODEL     │
│ Weight, Age,    │───┘    └──────────────┘
│ Breed (tabular) │
└─────────────────┘
```

**Option B: Separate Models per Breed**
Train specialized models for Angus, Hereford, etc.

**Option C: Features as Model Conditioning**
Weight/age help the model understand expected fat ranges.

---

## Technical Approach: Two-Stage Pipeline

### Stage 1: Instance Segmentation (Find & Isolate the Cow)

**Why**: Remove background noise (feeder rails, other cows, shadows) so the prediction model focuses only on the cow's body shape.

| Aspect | Details |
|--------|---------|
| **Model Type** | Instance Segmentation (YOLOv8-seg) |
| **Input** | Raw feeder image |
| **Output** | Pixel-perfect mask of cow's outline |
| **Annotation Type** | Polygon tracing around cow (~30-60 sec/image) |
| **Training Data** | 100-200 images with polygon annotations |

```
Raw Image                    Segmented Output
┌─────────────────────┐      ┌─────────────────────┐
│     ▓▓▓▓▓▓▓▓▓       │      │                     │
│     ▓▓▓COW▓▓▓       │ ──>  │     ▓▓▓▓▓▓▓▓▓       │
│     ▓▓▓▓▓▓▓▓▓       │      │     ▓▓▓▓▓▓▓▓▓       │
│   feeder rail       │      │     (black bg)      │
└─────────────────────┘      └─────────────────────┘
```

### Stage 2: P8 Regression (Predict the Score)

**Why**: Learn the relationship between cow body shape/condition and actual P8 fat measurement.

| Aspect | Details |
|--------|---------|
| **Model Type** | Regression (ResNet, EfficientNet, or custom CNN) |
| **Input** | Masked/cropped cow image from Stage 1 |
| **Output** | P8 value in mm (continuous number) |
| **Annotation Type** | CSV file: `image_filename, p8_mm` |
| **Training Data** | Same images as Stage 1, with P8 values from kill data |

**Alternative: Classification Approach**  
If regression proves difficult with limited data, group P8 into categories:
```
Class 0: "Lean"      (P8 < 6mm)
Class 1: "Light"     (P8 6-10mm)
Class 2: "Medium"    (P8 10-15mm)
Class 3: "Moderate"  (P8 15-22mm)
Class 4: "Fat"       (P8 > 22mm)
```

---

## Implementation Phases

### Phase 1: Data Collection & Preparation (Current)
- [ ] Capture ~100 images from feeder camera
- [ ] Match each image to animal via RFID timestamp
- [ ] Retrieve P8 scores from kill data
- [ ] Create master CSV: `filename, animal_id, p8_mm`

### Phase 2: Segmentation Model
- [ ] Create Roboflow account (free tier)
- [ ] Create project: **Instance Segmentation**
- [ ] Upload images
- [ ] Annotate with polygons (trace cow outline)
- [ ] Train YOLOv8-seg model
- [ ] Export as ONNX
- [ ] Test on OAK-1-W

### Phase 3: P8 Prediction Model
- [ ] Use segmentation model to create masked images
- [ ] Split data: 70% train, 20% validation, 10% test
- [ ] Train regression/classification model
- [ ] Evaluate accuracy (MAE for regression, accuracy for classification)
- [ ] Export and test

### Phase 4: Integration
- [ ] Combine both models in single OAK pipeline
- [ ] Live inference at feeder
- [ ] Display predicted P8 score

---

## Key Concepts Learned

| Term | Meaning |
|------|---------|
| **Deep Learning** | ML using neural networks with many layers |
| **Instance Segmentation** | Detecting objects AND their exact pixel boundaries |
| **Regression** | Predicting a continuous number (vs classification = categories) |
| **Annotation** | Labeling training data (metadata, not on images themselves) |
| **Train/Valid/Test Split** | Separate data to prevent overfitting and measure true performance |
| **ONNX** | Universal model format for deployment |
| **Inference** | Running a trained model on new data |

---

## Dataset Split Rationale

```
┌─────────────────────────────────────────────────────┐
│                 100 IMAGES                          │
├───────────────────────────┬───────────┬─────────────┤
│      TRAIN (70)           │ VALID (20)│  TEST (10)  │
│                           │           │             │
│  Model learns from these  │ Tune &    │ Final exam  │
│                           │ monitor   │ (once only) │
└───────────────────────────┴───────────┴─────────────┘
```

---

## Hardware

| Device | Use |
|--------|-----|
| **OAK-1-W** | Wide-angle camera for image capture & inference |
| **Platform** | RVC2 (limited memory, need optimized models) |

---

## Tools

| Tool | Purpose |
|------|---------|
| **Roboflow** | Annotation, training, model management |
| **DepthAI** | OAK camera SDK |
| **ONNX** | Model export format |

---

## Expected Challenges

1. **Limited Data (100 images)**: May need to expand to 500+ for reliable regression
2. **Variation in P8**: Wide range of values may require more samples per range
3. **Lighting Changes**: Different times of day at feeder
4. **Cow Positioning**: Some cows may not be perfectly aligned
5. **Segmentation Quality**: Poor masks will affect P8 prediction

---

## Success Metrics

| Metric | Target (Experiment) | Target (Production) |
|--------|---------------------|---------------------|
| Segmentation IoU | > 70% | > 85% |
| P8 MAE (Mean Absolute Error) | < 5mm | < 2mm |
| P8 Classification Accuracy | > 60% | > 85% |

---

## Notes & Decisions

- Using **segmentation** (polygon masks) instead of simple bounding boxes to isolate cow from background noise
- Starting with **100 images** for learning, will scale to thousands for production
- P8 values come from **kill data** matched via RFID
- Camera position is **fixed** (elevated at feeder) - consistent framing
- May start with **classification** (categories) if regression is too hard with limited data

---

## Next Steps

1. Gather 100 images with matched P8 data
2. Create Roboflow segmentation project
3. Annotate images with polygon outlines
4. Train segmentation model
5. Evaluate and iterate

---

## Resources

- Roboflow: https://app.roboflow.com/
- DepthAI Examples: `oak-examples/` folder
- OAK Documentation: https://docs.luxonis.com/
