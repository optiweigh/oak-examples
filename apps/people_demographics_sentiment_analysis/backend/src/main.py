import depthai as dai

from utils.pipeline import create_pipeline
from utils.arguments import initialize_argparser


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(serveFrontend=False)

device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name
print(f"Device: {device.getDeviceId()}  Platform: {platform}")

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888ps
)

if args.fps_limit is None:
    args.fps_limit = 2 if platform == "RVC2" else 6
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )

with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    outputs = create_pipeline(
        pipeline=pipeline,
        fps=args.fps_limit,
        platform=platform,
        frame_type=frame_type
    )

    monitor_node = outputs["monitor_node"]

    def get_faces_service(_=None):
        '''Returns latest face detections and face statistics to the frontend'''
        payload = getattr(monitor_node, "latest_payload", None)

        if not payload:
            return {
                "faces": [],
                "stats": {
                    "age": 0.0,
                    "males": 0.0,
                    "females": 0.0,
                    "emotions": {
                        "Happiness": 0.0,
                        "Neutral": 0.0,
                        "Surprise": 0.0,
                        "Anger": 0.0,
                        "Sadness": 0.0,
                        "Fear": 0.0,
                        "Disgust": 0.0,
                        "Contempt": 0.0,
                    },
                },
            }
        return payload

    visualizer.registerService("Get Faces", get_faces_service)

    visualizer.addTopic("Video", outputs["rgb_preview"], "images")
    visualizer.addTopic("Annotations", outputs["annotations"], "images")

    print("Pipeline created.")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        key = visualizer.waitKey(1)
        pipeline.processTasks()
        if key == ord("q"):
            print("Got q key. Exiting...")
            break
