from typing import List, Tuple
import depthai as dai

from depthai_nodes.message.gathered_data import GatheredData

from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.message import Keypoints


class Stage2CropToFullRemapNode(dai.node.HostNode):
    """
    Merge detections from each crop into one message and REMAP their coordinates
    from crop-normalized -> FULL-FRAME normalized (0..1 on original frame).

    Input (via link_args): GatherData bundle
      - msg.reference_data : ImgDetections/Extended (stage-1 faces; full-frame)
      - msg.gathered       : List[ImgDetectionsExtended] OR List[Keypoints] (stage-2 on each crop)

    Output:
      - self.out : Same message type as input (ImgDetectionsExtended or Keypoints),
                   with all items remapped to FULL FRAME (so default visualizer overlays on main "Video").
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _remap_rotated_rect_in_place(det, fx0: float, fy0: float, fw: float, fh: float) -> None:
        """
        Remap det.rotated_rect (center/size/angle) crop-normalized -> full-frame-normalized.
        Mutates `det` in place (safe: call on a copy).
        """
        rr = getattr(det, "rotated_rect", None) or getattr(det, "rr", None)
        if rr is None:
            return

        cx = getattr(rr, "cx", getattr(getattr(rr, "center", None), "x", None))
        cy = getattr(rr, "cy", getattr(getattr(rr, "center", None), "y", None))
        w = getattr(rr, "w", getattr(getattr(rr, "size", None), "width", None))
        h = getattr(rr, "h", getattr(getattr(rr, "size", None), "height", None))
        angle = float(getattr(rr, "angle", 0.0))

        if cx is None or cy is None or w is None or h is None:
            return

        cx_f = fx0 + float(cx) * fw
        cy_f = fy0 + float(cy) * fh
        w_f = float(w) * fw
        h_f = float(h) * fh

        det.rotated_rect = (cx_f, cy_f, w_f, h_f, angle)

    @staticmethod
    def _remap_keypoints_in_place(keypoints_list: List, fx0: float, fy0: float, fw: float, fh: float) -> None:
        """Remap a list of keypoints (each has .x/.y normalized to crop) -> full-frame-normalized."""
        if not keypoints_list:
            return
        for kp in keypoints_list:
            if not hasattr(kp, "x") or not hasattr(kp, "y"):
                continue
            kp.x = fx0 + float(kp.x) * fw
            kp.y = fy0 + float(kp.y) * fh

    def build(self, gathered_pair_out: dai.Node.Output) -> "Stage2CropToFullRemapNode":
        self.link_args(gathered_pair_out)
        return self

    def process(self, msg) -> None:
        if isinstance(msg, GatheredData):
            ref = msg.reference_data
            gathered = msg.gathered
        else:
            ref = None
            gathered = []

        if not gathered:
            empty = ImgDetectionsExtended()
            empty.setTimestamp(ref.getTimestamp())
            empty.setSequenceNum(ref.getSequenceNum())
            empty.setTransformation(ref.getTransformation())
            self.out.send(empty)
            return

        out = gathered[0].copy()
        merged_dets: List = []
        merged_kps: List = []
        merged_edges: List[Tuple[int, int]] = []

        faces = list(getattr(ref, "detections", [])) if ref is not None else []
        N = min(len(faces), len(gathered))

        for i in range(N):
            face = faces[i]
            crop_msg = gathered[i]

            fx0, fy0, fx1, fy1 = float(face.xmin), float(face.ymin), float(face.xmax), float(face.ymax)
            fw = max(1e-6, fx1 - fx0)
            fh = max(1e-6, fy1 - fy0)

            if isinstance(crop_msg, ImgDetectionsExtended):
                dets = getattr(crop_msg, "detections", None)
                if not dets:
                    continue
                for det in dets:
                    try:
                        d = det.copy()
                    except Exception:
                        d = det

                    self._remap_rotated_rect_in_place(d, fx0, fy0, fw, fh)

                    kps = getattr(d, "keypoints", None)
                    if kps:
                        self._remap_keypoints_in_place(kps, fx0, fy0, fw, fh)
                    merged_dets.append(d)

            elif isinstance(crop_msg, Keypoints):
                kps = getattr(crop_msg, "keypoints", None)
                if not kps:
                    continue

                idx_offset = len(merged_kps)
                for kp in kps:
                    try:
                        k = kp.copy()
                    except Exception:
                        k = kp
                    self._remap_keypoints_in_place([k], fx0, fy0, fw, fh)
                    merged_kps.append(k)

                detection_edges = list(getattr(crop_msg, "edges", [])) or []
                for a, b in detection_edges:
                    merged_edges.append((idx_offset + int(a), idx_offset + int(b)))

        if isinstance(out, ImgDetectionsExtended):
            out.detections = merged_dets
        elif isinstance(out, Keypoints):
            out.keypoints = merged_kps
            out.edges = merged_edges

        self.out.send(out)
