
import depthai as dai


class CreateBlackDetectionIfNoDetection(dai.node.HostNode):
    """"""

    def build(self, nn_output: dai.Node.Output) -> "CreateBlackDetectionIfNoDetection":
        self.link_args(nn_output)
        self.sendProcessingToPipeline(False)
        return self

    def process(self, nn_output: dai.Buffer) -> None:
        global SECOND
        assert isinstance(nn_output, dai.ImgDetections)
        if len(nn_output.detections) == 0:
            detection = dai.ImgDetection()
            detection.confidence = 0.9
            detection.label = 1
            detection.xmin = 0.99
            detection.xmax = 1.4
            detection.ymin = 0.99
            detection.ymax = 1.4
            detection.labelName = "Black"
            nn_output.detections = [detection]
        self.out.send(nn_output)
