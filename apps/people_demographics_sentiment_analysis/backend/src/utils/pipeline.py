from dataclasses import dataclass
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge, GatherData

from .annotation_node import AnnotatePeopleFaces
from .generate_script_content_multi import generate_script_multi
from .monitor_node import MonitorFacesNode
from .people_faces_join import PeopleFacesJoin
from .filter_n_largest_bboxes import FilterNLargestBBoxes


@dataclass
class PipelineOutputs:
    rgb_preview: dai.Node.Output
    annotations: dai.Node.Output
    monitor_node: MonitorFacesNode


def create_pipeline(pipeline: dai.Pipeline, fps: int, platform: str, frame_type: dai.ImgFrame.Type) -> PipelineOutputs:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput(size=(1280, 720), type=frame_type, fps=fps)
    cam_out_encode = cam.requestOutput(size=(1280, 720), type=dai.ImgFrame.Type.NV12, fps=fps)

    # Face detection model
    face_nn_archive = get_nn_archive(model="luxonis/yunet:640x360", platform=platform)
    face_nn_w, face_nn_h = face_nn_archive.getInputSize()

    # Emotion recognition model
    emotions_nn_archive = get_nn_archive(model="luxonis/emotion-recognition:260x260", platform=platform)
    emotions_nn_w, emotions_nn_h = emotions_nn_archive.getInputSize()

    # Age-gender recognition model
    age_gender_nn_archive = get_nn_archive(model="luxonis/age-gender-recognition:62x62", platform=platform)
    age_gender_nn_w, age_gender_nn_h = age_gender_nn_archive.getInputSize()

    # People detector
    people_nn_archive = get_nn_archive(model="luxonis/yolov6-nano:r2-coco-512x288", platform=platform)
    people_nn_w, people_nn_h = people_nn_archive.getInputSize()

    # Reidentification
    re_id_nn_archive = get_nn_archive(model="luxonis/arcface:lfw-112x112", platform=platform)
    re_id_nn_w, re_id_nn_h = re_id_nn_archive.getInputSize()

    script = pipeline.create(dai.node.Script)

    # Resize input frames to NN models input sizes
    img_manip_face = pipeline.create(dai.node.ImageManip)
    img_manip_face.initialConfig.setOutputSize(face_nn_w, face_nn_h)
    img_manip_face.initialConfig.setReusePreviousImage(False)
    img_manip_face.inputImage.setBlocking(True)
    cam_out.link(img_manip_face.inputImage)

    img_manip_people = pipeline.create(dai.node.ImageManip)
    img_manip_people.initialConfig.setOutputSize(people_nn_w, people_nn_h)
    img_manip_people.initialConfig.setFrameType(frame_type)
    img_manip_people.initialConfig.setReusePreviousImage(False)
    img_manip_people.inputImage.setBlocking(True)
    cam_out.link(img_manip_people.inputImage)

    img_manip_emotions = pipeline.create(dai.node.ImageManip)
    img_manip_emotions.inputConfig.setWaitForMessage(True)
    img_manip_emotions.inputImage.setBlocking(True)

    img_manip_age_gender = pipeline.create(dai.node.ImageManip)
    img_manip_age_gender.inputConfig.setWaitForMessage(True)
    img_manip_age_gender.inputImage.setBlocking(True)

    img_manip_re_id = pipeline.create(dai.node.ImageManip)
    img_manip_re_id.inputConfig.setWaitForMessage(True)
    img_manip_re_id.inputImage.setBlocking(True)

    script.outputs["manip_cfg_emotions"].link(img_manip_emotions.inputConfig)
    script.outputs["manip_img_emotions"].link(img_manip_emotions.inputImage)

    script.outputs["manip_cfg_age_gender"].link(img_manip_age_gender.inputConfig)
    script.outputs["manip_img_age_gender"].link(img_manip_age_gender.inputImage)

    script.outputs["manip_cfg_reid"].link(img_manip_re_id.inputConfig)
    script.outputs["manip_img_reid"].link(img_manip_re_id.inputImage)

    face_nn = pipeline.create(ParsingNeuralNetwork).build(img_manip_face.out, face_nn_archive)
    emotions_nn = pipeline.create(ParsingNeuralNetwork).build(img_manip_emotions.out, emotions_nn_archive)
    age_gender_nn = pipeline.create(ParsingNeuralNetwork).build(img_manip_age_gender.out, age_gender_nn_archive)
    re_id_nn = pipeline.create(ParsingNeuralNetwork).build(img_manip_re_id.out, re_id_nn_archive)
    people_det = pipeline.create(dai.node.DetectionNetwork).build(img_manip_people.out, people_nn_archive)

    people_det.setConfidenceThreshold(0.65)
    face_nn.getParser().setConfidenceThreshold(0.85)
    face_nn.getParser().setIOUThreshold(0.75)

    # Filter face detections - limit processed face crops to n_face_crops
    filtered_face_det = pipeline.create(FilterNLargestBBoxes).build(
        face_detections=face_nn.out,
        n_face_crops=3,
    )

    # TODO: remove once Script node works with ImgDetectionsExtended
    face_det = pipeline.create(ImgDetectionsBridge).build(filtered_face_det.out)

    script_code = generate_script_multi(
        emotions_nn_w, emotions_nn_h,
        age_gender_nn_w, age_gender_nn_h,
        re_id_nn_w, re_id_nn_h
    )
    script.setScript(script_code)
    face_det.out.link(script.inputs["det_in"])
    cam_out.link(script.inputs["preview"])

    # Detections and recognitions sync
    gather_data_re_id = pipeline.create(GatherData).build(camera_fps=fps)
    re_id_nn.out.link(gather_data_re_id.input_data)
    filtered_face_det.out.link(gather_data_re_id.input_reference)

    gather_data_emotions = pipeline.create(GatherData).build(camera_fps=fps)
    emotions_nn.out.link(gather_data_emotions.input_data)
    filtered_face_det.out.link(gather_data_emotions.input_reference)

    gather_data_age_gender = pipeline.create(GatherData).build(camera_fps=fps)
    age_gender_nn.outputs.link(gather_data_age_gender.input_data)
    filtered_face_det.out.link(gather_data_age_gender.input_reference)

    gather_data_crops = pipeline.create(GatherData).build(camera_fps=fps)
    img_manip_emotions.out.link(gather_data_crops.input_data)
    filtered_face_det.out.link(gather_data_crops.input_reference)

    # Object tracker
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setDetectionLabelsToTrack([0])                  # 0 for people
    tracker.setTrackerThreshold(0.5)
    tracker.setTrackletMaxLifespan(15)
    tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    people_det.passthrough.link(tracker.inputTrackerFrame)
    people_det.passthrough.link(tracker.inputDetectionFrame)
    people_det.out.link(tracker.inputDetections)

    # Merge face detections with people detections
    people_faces_join = pipeline.create(PeopleFacesJoin).build(
        track=tracker.out,
        age_gender=gather_data_age_gender.out,
        emotions=gather_data_emotions.out,
        reid=gather_data_re_id.out,
        crops=gather_data_crops.out,
    )

    # Faces to display on crop panel
    monitor = pipeline.create(MonitorFacesNode).build(people_faces_join.out)

    annotation_node = pipeline.create(AnnotatePeopleFaces).build(people_faces_join.out)

    h264_encoder = pipeline.create(dai.node.VideoEncoder)
    h264_encoder.setDefaultProfilePreset(
        fps=fps, profile=dai.VideoEncoderProperties.Profile.H264_MAIN
    )
    cam_out_encode.link(h264_encoder.input)

    return PipelineOutputs(
        rgb_preview=h264_encoder.out,
        annotations=annotation_node.out,
        monitor_node=monitor,
    )


def get_nn_archive(model: str, platform: str):
    description = dai.NNModelDescription(model=model, platform=platform)
    archive = dai.NNArchive(dai.getModelFromZoo(description))

    return archive
