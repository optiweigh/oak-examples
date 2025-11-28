import depthai as dai

from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge
from .filter_n_largest_bboxes import FilterNLargestBBoxes


class FaceDetectionStage:
    """
    Builds the face detection pipeline segment:
      ImageManip -> ParsingNeuralNetwork -> (filter top N faces) -> ImgDetectionsBridge
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        image_source: dai.Node.Output,
        archive: dai.NNArchive,
        max_faces: int = 3,
    ):
        self._pipeline = pipeline
        self._image_source = image_source
        self._archive = archive
        self._max_faces = max_faces

        self._model_w = archive.getInputWidth()
        self._model_h = archive.getInputHeight()

        self._nn: ParsingNeuralNetwork = None
        self._filtered_face_det: FilterNLargestBBoxes = None
        self._filtered_face_det_bridge: ImgDetectionsBridge = None

    def build(self) -> "FaceDetectionStage":
        img_manip_face = self._create_image_manip()

        self._nn = self._pipeline.create(ParsingNeuralNetwork).build(
            img_manip_face.out, self._archive
        )
        self._nn.getParser().setConfidenceThreshold(0.85)
        self._nn.getParser().setIOUThreshold(0.75)

        self._filtered_face_det = self._pipeline.create(FilterNLargestBBoxes).build(
            face_detections=self.nn.out,
            n_face_crops=self._max_faces,
        )
        self._filtered_face_det_bridge = self._pipeline.create(
            ImgDetectionsBridge
        ).build(self._filtered_face_det.out)

        return self

    def _create_image_manip(self) -> dai.node.ImageManip:
        img_manip_face = self._pipeline.create(dai.node.ImageManip)
        img_manip_face.initialConfig.setOutputSize(self._model_w, self._model_h)
        img_manip_face.initialConfig.setReusePreviousImage(False)
        img_manip_face.inputImage.setBlocking(True)
        self._image_source.link(img_manip_face.inputImage)
        return img_manip_face

    @property
    def nn(self) -> ParsingNeuralNetwork:
        if self._nn is None:
            raise RuntimeError("FaceNN.build() must be called before accessing nn.")
        return self._nn

    @property
    def filtered_output(self) -> dai.Node.Output:
        if self._filtered_face_det is None:
            raise RuntimeError(
                "FaceNN.build() must be called before accessing filtered_output."
            )
        return self._filtered_face_det.out

    @property
    def filtered_bridge_output(self) -> dai.Node.Output:
        if self._filtered_face_det_bridge is None:
            raise RuntimeError(
                "FaceNN.build() must be called before accessing filtered_bridge_output."
            )
        return self._filtered_face_det_bridge.out
