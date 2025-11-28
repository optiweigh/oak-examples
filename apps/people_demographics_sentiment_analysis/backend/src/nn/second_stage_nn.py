from typing import Optional
import depthai as dai

from depthai_nodes.node import ParsingNeuralNetwork, GatherData


class SecondStageNN:
    """
    Builds a second-stage neural network block that takes image crops and produces NN outputs aligned with reference detections.

    Receives detection crops from the first stage and runs them through:
        -> ImageManip (resize to NN input resolution);
        -> ParsingNeuralNetwork (run the model);
        -> GatherData (synchronize NN outputs with the correct reference detections).

    Parameters
    ----------
    pipeline : dai.Pipeline
    img_source : dai.Node.Output
        Face crops from FaceCropsStage.
    archive : dai.NNArchive
    multi_head_nn : bool
        Whether the NN has multiple output heads or a single default output.
    camera_fps : int
    reference_detections : dai.Node.Output
        The reference detections stream from the first-stage detector.
        GatherData uses this to align NN predictions to the correct detection.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        img_source: dai.Node.Output,
        archive: dai.NNArchive,
        multi_head_nn: bool,
        camera_fps: int,
        reference_detections: dai.Node.Output,
    ):
        self._pipeline = pipeline
        self._img_source = img_source
        self._archive = archive
        self._camera_fps = camera_fps
        self._reference_detections = reference_detections
        self._multi_head_nn = multi_head_nn

        self._model_w = archive.getInputWidth()
        self._model_h = archive.getInputHeight()

        self._img_manip: Optional[dai.node.ImageManip] = None
        self._nn: Optional[ParsingNeuralNetwork] = None
        self._gather_data: Optional[GatherData] = None

    def build(self) -> "SecondStageNN":
        self._img_manip = self._create_image_manip()
        self._nn = self._pipeline.create(ParsingNeuralNetwork).build(
            self._img_manip.out, self._archive
        )
        self._gather_data = self._create_gather_data()

        return self

    def _create_image_manip(self) -> dai.node.ImageManip:
        img_manip = self._pipeline.create(dai.node.ImageManip)
        img_manip.initialConfig.setOutputSize(self._model_w, self._model_h)
        img_manip.inputImage.setBlocking(True)
        img_manip.setMaxOutputFrameSize(self._model_w * self._model_h * 3)
        self._img_source.link(img_manip.inputImage)

        return img_manip

    def _create_gather_data(self) -> GatherData:
        gather_data = self._pipeline.create(GatherData).build(
            camera_fps=self._camera_fps
        )
        if self._multi_head_nn:
            self._nn.outputs.link(gather_data.input_data)
        else:
            self._nn.out.link(gather_data.input_data)
        self._reference_detections.link(gather_data.input_reference)

        return gather_data

    @property
    def synced_out(self) -> dai.Node.Output:
        """NN output aligned to reference face detections."""
        return self._gather_data.out
