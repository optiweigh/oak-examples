import depthai as dai

from utils.pipeline import create_pipeline, PipelineOutputs
from utils.arguments import initialize_argparser


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(serveFrontend=False)

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Device: {device.getDeviceId()}  Platform: {platform}")

if platform != "RVC4":
    raise ValueError("This example is supported only on RVC4 platform")

frame_type = dai.ImgFrame.Type.BGR888i

if args.fps_limit is None:
    args.fps_limit = 15
    print(
        f"\nFPS limit set to {args.fps_limit}. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    outputs: PipelineOutputs = create_pipeline(
        pipeline=pipeline,
        fps=args.fps_limit,
        platform=platform,
        frame_type=frame_type,
    )

    visualizer.registerService("Get Faces", outputs.monitor_node.visualizer_get_faces_payload)

    visualizer.addTopic("Video", outputs.rgb_preview, "images")
    visualizer.addTopic("Annotations", outputs.annotations, "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        pipeline.processTasks()
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
