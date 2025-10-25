
import sys
from pathlib import Path

# --- Project path & config (match your original script’s root) ---
DEPTHAI_NODES_FOLDER = Path("/Users/petrnovota/programming_projects/depthai-nodes")
sys.path.insert(0, str(DEPTHAI_NODES_FOLDER))  # <— keep this so imports work from anywhere

import depthai as dai
import host_nodes
import pipeline_helpers as pipe
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
from depthai_nodes.node.extended_neural_network import ExtendedNeuralNetwork

from utils.coordinate_transformer import transform_ymin, transform_ymax
from utils.arguments import initialize_argparser
from utils.pipeline_build_helpers import build_h264_stream


_, args = initialize_argparser()
args.device = "1631075878"

HIGH_RES_WIDTH, HIGH_RES_HEIGHT = 2000, 2000
LOW_RES_WIDTH, LOW_RES_HEIGHT = 640, 640
PEOPLE_DETECTION_MODEL = "luxonis/scrfd-person-detection:25g-640x640"
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

frame_type = dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
if not args.fps_limit:
    args.fps_limit = 8 if platform == "RVC2" else 15

with dai.Pipeline(device) as pipeline:
    rgb_low_res_out = pipe.create_rgb(
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
        enable_tiling=True,
        input_size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
    )
    face_detection_nn.setTilingGridSize((4,4))
    largest_face_detection = host_nodes.PickLargestBbox().build(face_detection_nn.out)
    face_detection_1_stage_nn_as_img_det = host_nodes.SafeImgDetectionsExtendedBridge().build(largest_face_detection.out)
    face_detection_1_stage_nn_a_as_img_det = host_nodes.CreateBlackDetectionIfNoDetection().build(face_detection_1_stage_nn_as_img_det.out)

    rgb_low_res_encoder = build_h264_stream(
        pipeline,
        src=rgb_low_res_out,
        size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        assume_nv12=False,
    )

    visualizer.addTopic("640x640 RGB", rgb_low_res_encoder, "low_res_image")
    visualizer.addTopic("NN detections", face_detection_nn.out, "face detections")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
