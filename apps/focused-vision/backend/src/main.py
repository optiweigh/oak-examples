import logging

import depthai as dai

import pipeline_builders
from arguments import initialize_argparser


logger = logging.getLogger(__name__)

_, args = initialize_argparser()
args.device = "1631075878"

logger.error(f"Starting")
frame_type = dai.ImgFrame.Type.BGR888i
HIGH_RES_WIDTH, HIGH_RES_HEIGHT = 2000, 2000
LOW_RES_WIDTH, LOW_RES_HEIGHT = 640, 640
PEOPLE_DETECTION_MODEL = "luxonis/scrfd-person-detection:25g-640x640"
FACE_DETECTION_MODEL = "luxonis/yunet:320x240"
if not args.fps_limit:
    args.fps_limit = 13

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

if platform == "RVC2":
    logger.error(f"Detected platform is {platform}. Application can only run with RVC4. Exiting.")
    exit(1)

with dai.Pipeline(device) as pipeline:
    rgb_low_res_out, rgb_high_res_out = pipeline_builders.build_rgb(
        pipeline=pipeline,
        high_res=(HIGH_RES_WIDTH, HIGH_RES_HEIGHT),
        low_res=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
    )
    rgb_low_res_encoder = pipeline_builders.build_h264_stream(
        pipeline,
        src=rgb_low_res_out,
        size=(LOW_RES_WIDTH, LOW_RES_HEIGHT),
        fps=args.fps_limit,
        profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
        assume_nv12=False,
    )
    # Naive face detection
    face_detection_naive, face_crops_naive = pipeline_builders.build_naive_approach(
        pipeline=pipeline,
        rgb_low_res_out=rgb_low_res_out,
        face_detection_model_name=FACE_DETECTION_MODEL,
        frame_type=frame_type,
    )

    # 2-stage face detection
    face_crops_2_stage = pipeline_builders.build_2_stage_face_detection(
        pipeline=pipeline,
        rgb_low_res_out=rgb_low_res_out,
        rgb_high_res_out=rgb_high_res_out,
        people_detection_model_name=PEOPLE_DETECTION_MODEL,
        face_detection_model_name=FACE_DETECTION_MODEL,
        frame_type=frame_type,
        fps_limit=args.fps_limit,
    )

    # 1 stage with tiling
    face_crops_tiling = pipeline_builders.build_1_stage_with_tiling(
        pipeline=pipeline,
        rgb_high_res_out=rgb_high_res_out,
        high_res_width=HIGH_RES_WIDTH,
        high_res_height=HIGH_RES_HEIGHT,
        face_detection_model_name=FACE_DETECTION_MODEL,
        frame_type=frame_type,
    )

    visualizer.addTopic("640x640 RGB", rgb_low_res_encoder, "low_res_image")
    visualizer.addTopic("NN detections", face_detection_naive.out, "low_res_image")
    # visualizer.addTopic("People detections", people_detection.out, "low_res_image")
    visualizer.addTopic("Non-Focus Head Crops", face_crops_naive.out, "non_focus_head_crops")
    visualizer.addTopic("Focused Vision Head Crops", face_crops_2_stage.out, "focused_vision_head_crops")
    visualizer.addTopic("Focused with Tiling", face_crops_tiling.out, "focused_vision_tiling")
    logger.error(f"Starting Pipeline")
    pipeline.start()
    visualizer.registerPipeline(pipeline)

    counter = 0
    while pipeline.isRunning():
        pipeline.processTasks()
        key = visualizer.waitKey(1)
        counter += 1
        if counter % 100_000 == 0:
            logger.error(f"Running")
        if key == ord("q"):
            print("[MAIN] Got q key. Exiting...")
            break
