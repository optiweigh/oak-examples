import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.utils import AnnotationHelper


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self._padding = 0.0  # padding in normalized units (0..1)

    def build(self, gathered_pair_out: dai.Node.Output, padding: float = 0.0) -> "AnnotationNode":
        self._padding = float(padding)
        self.link_args(gathered_pair_out)
        return self

    def process(self, gather_msg) -> None:
        # We only need the stage-1 detections
        dets_msg: ImgDetectionsExtended = gather_msg.reference_data
        ann = AnnotationHelper()

        if not dets_msg or not dets_msg.detections:
            out = ann.build(timestamp=dets_msg.getTimestamp(), sequence_num=dets_msg.getSequenceNum())
            self.out.send(out)
            return

        pad = self._padding

        for face_det in dets_msg.detections:
            rect = face_det.rotated_rect
            # axis-aligned box around the (possibly rotated) rect, with optional padding
            w = rect.size.width
            h = rect.size.height
            new_w = w + 2.0 * pad
            new_h = h + 2.0 * pad

            left   = rect.center.x - new_w / 2.0
            top    = rect.center.y - new_h / 2.0
            right  = rect.center.x + new_w / 2.0
            bottom = rect.center.y + new_h / 2.0

            # clamp to [0, 1]
            left   = max(0.0, min(1.0, left))
            top    = max(0.0, min(1.0, top))
            right  = max(0.0, min(1.0, right))
            bottom = max(0.0, min(1.0, bottom))

            if right > left and bottom > top:
                ann.draw_rectangle([left, top], [right, bottom])

        out = ann.build(timestamp=dets_msg.getTimestamp(), sequence_num=dets_msg.getSequenceNum())
        self.out.send(out)
