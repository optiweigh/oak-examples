import depthai as dai

from config.arguments import initialize_argparser
from config.system_configuration import SystemConfiguration

from camera.camera_source import CameraSource
from nn.nn_archives_provider import NNArchiveProvider

from faces.face_detection import FaceDetectionStage
from faces.cropping.face_crops_stage import FaceCropsStage
from nn.second_stage_nn import SecondStageNN

from faces.features.face_features_node import FaceFeaturesNode

from people.people_tracking import PeopleTrackingStage
from people.joiner.people_faces_join import PeopleJoinNode

from visualization.monitor_node import MonitorFacesNode
from visualization.annotation_node import AnnotationNode


def main():
    _, args = initialize_argparser()
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    visualizer = dai.RemoteConnection(serveFrontend=False)

    platform = device.getPlatform().name
    print(f"Device: {device.getDeviceId()}  Platform: {platform}")

    if platform != "RVC4":
        raise ValueError("This example is supported only on RVC4 platform")

    config = SystemConfiguration(platform, args).build()
    video_cfg = config.get_video_config()
    model_names = config.get_model_names()

    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline...")

        camera_source = CameraSource(pipeline=pipeline, cfg=video_cfg).build()
        cam_out = camera_source.preview
        encoded_out = camera_source.encoded

        archives = NNArchiveProvider(platform=platform, models=model_names)

        face_det = FaceDetectionStage(
            pipeline=pipeline, image_source=cam_out, archive=archives.face()
        ).build()

        face_crops = FaceCropsStage(
            pipeline=pipeline,
            preview_source=cam_out,
            detections_source=face_det.filtered_bridge_output,
            face_reference_detections=face_det.filtered_output,
            camera_fps=video_cfg.fps,
        ).build()

        # Emotions
        emotions_stage = SecondStageNN(
            pipeline=pipeline,
            img_source=face_crops.out,
            archive=archives.emotions(),
            multi_head_nn=False,
            camera_fps=video_cfg.fps,
            reference_detections=face_det.filtered_output,
        ).build()

        # Age/gender
        age_gender_stage = SecondStageNN(
            pipeline=pipeline,
            img_source=face_crops.out,
            archive=archives.age_gender(),
            multi_head_nn=True,
            camera_fps=video_cfg.fps,
            reference_detections=face_det.filtered_output,
        ).build()

        # Re-ID
        reid_stage = SecondStageNN(
            pipeline=pipeline,
            img_source=face_crops.out,
            archive=archives.reid(),
            multi_head_nn=False,
            camera_fps=video_cfg.fps,
            reference_detections=face_det.filtered_output,
        ).build()

        # Object tracker
        tracker = PeopleTrackingStage(
            pipeline=pipeline,
            image_source=cam_out,
            fps=video_cfg.fps,
            archive=archives.people(),
        ).build()

        face_features_node = pipeline.create(FaceFeaturesNode).build(
            age_gender=age_gender_stage.synced_out,
            emotions=emotions_stage.synced_out,
            reid=reid_stage.synced_out,
            crops=face_crops.synced_out,
        )

        people_join_node = pipeline.create(PeopleJoinNode).build(
            faces=face_features_node.out,
            tracklets=tracker.out,
        )

        # Faces to display on crop panel
        monitor_node = pipeline.create(MonitorFacesNode).build(people_join_node.out)

        annotation_node = pipeline.create(AnnotationNode).build(people_join_node.out)

        visualizer.registerService("Get Faces", monitor_node.visualizer_get_payload)

        visualizer.addTopic("Video", encoded_out, "images")
        visualizer.addTopic("Annotations", annotation_node.out, "images")

        print("Pipeline created.")

        pipeline.start()
        visualizer.registerPipeline(pipeline)

        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            pipeline.processTasks()
            if key == ord("q"):
                print("Got q key. Exiting...")
                break


if __name__ == "__main__":
    main()
