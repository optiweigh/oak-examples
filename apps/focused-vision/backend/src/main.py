
import sys
from pathlib import Path
import logging

# --- Project path & config (match your original script’s root) ---
# DEPTHAI_NODES_FOLDER = Path("/Users/petrnovota/programming_projects/depthai-nodes")
# sys.path.insert(0, str(DEPTHAI_NODES_FOLDER))  # <— keep this so imports work from anywhere

import depthai as dai
import host_nodes
import pipeline_helpers as pipe
from depthai_nodes.node.extended_neural_network import ExtendedNeuralNetwork
from depthai_nodes.node.stage_2_neural_network import Stage2NeuralNetwork

from utils.coordinate_transformer import transform_ymin, transform_ymax
from utils.arguments import initialize_argparser
from utils.pipeline_build_helpers import build_h264_stream

logger = logging.getLogger(__name__)

_, args = initialize_argparser()
args.device = "1631075878"

logger.error(f"StartingIII")
HIGH_RES_WIDTH, HIGH_RES_HEIGHT = 2000, 2000
LOW_RES_WIDTH, LOW_RES_HEIGHT = 640, 640
PEOPLE_DETECTION_MODEL = "luxonis/scrfd-person-detection:25g-640x640"
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
if not args.fps_limit:
    args.fps_limit = 5

with dai.Pipeline(device) as pipeline:
    rgb_low_res_out, rgb_high_res_out = pipe.create_rgb(
        pipeline=pipeline,
        high_res=(HIGH_RES_WIDTH, HIGH_RES_HEIGHT),
        low_res=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
    )
    # 1-stage face detection
    face_detection_nn = pipeline.create(ExtendedNeuralNetwork)
    face_detection_nn.build(
        input=rgb_low_res_out,
        input_resize_mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        nn_source=FACE_DETECTION_MODEL,
        enable_detection_filtering=True,
        # enable_tiling=True,
        # input_size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
    )
    # face_detection_nn.setTilingGridSize((4,4))
    largest_face_detection = host_nodes.PickLargestBbox().build(face_detection_nn.out)
    face_detection_1_stage_nn_as_img_det = host_nodes.SafeImgDetectionsExtendedBridge().build(largest_face_detection.out)
    face_detection_1_stage_nn_as_img_det.ignore_rotation()
    router_non_focused = host_nodes.Router().build(face_detection_1_stage_nn_as_img_det.out, rgb_low_res_out)
    head_cropper_low_res = pipe.build_roi_cropper(
        pipeline=pipeline,
        preview_stream=router_non_focused.rgb,
        det_stream=router_non_focused.has_detections,
        out_size=(320, 320),
        frame_type=frame_type,
        padding=0.02,
        pool_size=5,
        cfg_queue_size=5,
    )
    black_image_generator_non_focused = host_nodes.BlackFrame().build(router_non_focused.no_detections)
    head_crops_non_focused = host_nodes.Merger().build(head_cropper_low_res, black_image_generator_non_focused.out)

    # 2-stage face detection
    people_detection_nn = pipeline.create(ExtendedNeuralNetwork)
    people_detection_nn.build(
        input=rgb_low_res_out,
        input_resize_mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        nn_source=PEOPLE_DETECTION_MODEL,
        enable_detection_filtering=True,
        # enable_tiling=True,
        # input_size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
    )
    largest_people_detection = host_nodes.PickLargestBbox().build(people_detection_nn.out)
    largest_people_detection_as_img_detection = host_nodes.SafeImgDetectionsExtendedBridge().build(largest_people_detection.out)
    largest_people_detection_as_img_detection.ignore_rotation()
    largest_people_detection_cropped = host_nodes.CropPersonDetectionWaistDown(
        ymin_transformer=transform_ymin,
        ymax_transformer=transform_ymax,
    ).build(largest_people_detection_as_img_detection.out)
    face_people_gathered = pipeline.create(Stage2NeuralNetwork).build(
        img_frame=rgb_high_res_out,
        stage_1_nn=largest_people_detection_cropped.out,
        nn_source=FACE_DETECTION_MODEL,
        input_resize_mode=dai.ImageManipConfig.ResizeMode.LETTERBOX,
        fps=args.fps_limit,
        remap_detections=True,
    )
    face_detection = host_nodes.FaceDetectionFromGatheredData().build(face_people_gathered.out)
    face_detection_as_img_detection = host_nodes.SafeImgDetectionsExtendedBridge().build(face_detection.out)
    router = host_nodes.Router().build(face_detection_as_img_detection.out, rgb_high_res_out)
    head_cropper_high_res = pipe.build_roi_cropper(
        pipeline=pipeline,
        preview_stream=router.rgb,
        det_stream=router.has_detections,
        out_size=(320, 320),
        frame_type=frame_type,
        padding=0.02,
        pool_size=7,
        cfg_queue_size=5,
    )
    black_image_generator = host_nodes.BlackFrame().build(router.no_detections)
    head_crops_focused = host_nodes.Merger().build(head_cropper_high_res, black_image_generator.out)

    rgb_low_res_encoder = build_h264_stream(
        pipeline,
        src=rgb_low_res_out,
        size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        assume_nv12=False,
    )

    visualizer.addTopic("640x640 RGB", rgb_low_res_encoder, "low_res_image")
    visualizer.addTopic("NN detections", face_detection_nn.out, "face_detections")
    visualizer.addTopic("Non-Focus Head Crops", head_crops_non_focused.out, "non_focus_head_crops")
    visualizer.addTopic("Focused Vision Head Crops", head_crops_focused.out, "focused_vision_head_crops")
    logger.error(f"Starting Pipeline")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    counter = 0
    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        counter += 1
        if counter % 2_000 == 0:
            logger.error(f"Running")
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
