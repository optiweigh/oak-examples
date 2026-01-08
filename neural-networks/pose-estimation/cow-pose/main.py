"""
Cow Pose Estimation Example

This example combines:
1. Wildlife Megadetector: Trained on camera trap images, handles various angles
   (front-on, top-down, profile views)
2. SuperAnimal Landmarker: Estimates pose/keypoints on detected animals
3. Snapshot capture: Saves clear photos when cows are detected

Key ML Concepts Used:
- Object Detection: Finding objects in an image and drawing bounding boxes
- Camera Trap Data: Wildlife Megadetector handles varied angles/lighting
- Two-stage pipeline: First detect, then analyze pose on cropped regions
- Inference: Running trained models on new images to get predictions
"""

from pathlib import Path

import depthai as dai
from depthai_nodes.node import (
    ParsingNeuralNetwork,
    ImgDetectionsBridge,
    ImgDetectionsFilter,
    GatherData,
)
from depthai_nodes.node.utils import generate_script_content

from utils.arguments import initialize_argparser
from utils.annotation_node import AnnotationNode

# Configuration
PADDING = 0.1  # Extra padding around detected cows for pose estimation

# Wildlife Megadetector class 0 = "animal" (detects all animals, any angle)
# This model is trained on camera trap images which include:
# - Front-on views, top-down/back views, profile/side views
# - Various lighting conditions and partial occlusions
ANIMAL_CLASS_ID = 0

_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Platform: {platform}")

frame_type = (
    dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
)

if args.fps_limit is None:
    args.fps_limit = 5 if platform == "RVC2" else 20
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    # ========== STAGE 1: Wildlife Megadetector ==========
    # Wildlife Megadetector is trained on camera trap images
    # which include many different viewing angles:
    # - Front-on, back/top-down, profile views
    # - Various lighting conditions
    # - Partial occlusions
    # Classes: 0=animal, 1=person, 2=vehicle
    
    det_model_description = dai.NNModelDescription.fromYamlFile(
        f"wildlife_megadetector.{platform}.yaml"
    )
    det_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # ========== STAGE 2: SuperAnimal Pose Estimation ==========
    # Once we detect a cow, we crop that region and estimate its pose
    # Pose = positions of body parts (head, legs, tail, etc.)
    
    pose_model_description = dai.NNModelDescription.fromYamlFile(
        f"superanimal_landmarker.{platform}.yaml"
    )
    pose_nn_archive = dai.NNArchive(dai.getModelFromZoo(pose_model_description))
    pose_model_w, pose_model_h = pose_nn_archive.getInputSize()

    # ========== Camera/Video Input ==========
    if args.media_path:
        replay = pipeline.create(dai.node.ReplayVideo)
        replay.setReplayVideoFile(Path(args.media_path))
        replay.setOutFrameType(dai.ImgFrame.Type.NV12)
        replay.setLoop(True)
    else:
        cam = pipeline.create(dai.node.Camera).build()
    input_node = replay if args.media_path else cam
    
    # High-resolution output for snapshots (1080p)
    if args.media_path:
        hires_output = replay.out
    else:
        hires_output = cam.requestOutput((1920, 1080), type=frame_type)

    # ========== Detection Neural Network ==========
    # ParsingNeuralNetwork = runs the model AND parses its output into detections
    detection_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        input_node, det_nn_archive, fps=args.fps_limit
    )

    # Wildlife Megadetector classes: 0=animal, 1=person, 2=vehicle
    # We filter to only keep animals (class 0)
    classes = det_nn_archive.getConfig().model.heads[0].metadata.classes
    print(f"Wildlife Megadetector classes: {classes}")
    print("Filtering to keep only 'animal' detections (class 0)")
    print("This model handles front-on, top-down, and profile views!")

    # Filter to only keep animal detections
    # This removes people and vehicles
    detections_filter = pipeline.create(ImgDetectionsFilter).build(
        detection_nn.out, labels_to_keep=[ANIMAL_CLASS_ID]
    )

    # ========== Crop Detected Cows for Pose Estimation ==========
    # For each detected cow, we crop that region and resize it
    # to feed into the pose estimation model
    
    script = pipeline.create(dai.node.Script)
    detections_filter.out.link(script.inputs["det_in"])
    detection_nn.passthrough.link(script.inputs["preview"])
    script_content = generate_script_content(
        resize_width=pose_model_w,
        resize_height=pose_model_h,
        padding=PADDING,
        resize_mode="STRETCH",
    )
    script.setScript(script_content)

    pose_manip = pipeline.create(dai.node.ImageManip)
    pose_manip.initialConfig.setOutputSize(pose_model_w, pose_model_h)
    pose_manip.inputConfig.setWaitForMessage(True)

    script.outputs["manip_cfg"].link(pose_manip.inputConfig)
    script.outputs["manip_img"].link(pose_manip.inputImage)

    # ========== Pose Estimation Neural Network ==========
    pose_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        pose_manip.out, pose_nn_archive
    )

    # Bridge converts detection format for synchronization
    detections_bridge = pipeline.create(ImgDetectionsBridge).build(
        detections_filter.out
    )

    # ========== Synchronize Detections with Pose Results ==========
    # GatherData matches each detection with its pose estimation result
    # This is needed because pose estimation runs on cropped images
    gather_data = pipeline.create(GatherData).build(args.fps_limit)
    detections_bridge.out.link(gather_data.input_reference)
    pose_nn.out.link(gather_data.input_data)

    # ========== Annotation Node (draws results + saves snapshots) ==========
    connection_pairs = (
        pose_nn_archive.getConfig()
        .model.heads[0]
        .metadata.extraParams["skeleton_edges"]
    )
    annotation_node = pipeline.create(AnnotationNode).build(
        input_detections=gather_data.out,
        connection_pairs=connection_pairs,
        padding=PADDING,
        video_frame=hires_output,
        snapshot_cooldown=2.0,
        blur_threshold=100.0,
        confidence_threshold=0.7,  # Only save snapshots for detections > 70% confidence
    )

    # ========== Visualization ==========
    visualizer.addTopic("Video", detection_nn.passthrough, "images")
    visualizer.addTopic("Detections", annotation_node.out_detections, "images")
    visualizer.addTopic("Pose", annotation_node.out_pose_annotations, "images")

    print("Pipeline created.")
    print("Looking for ANIMALS using Wildlife Megadetector")
    print("Handles: front-on, top-down, back, and profile views")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
