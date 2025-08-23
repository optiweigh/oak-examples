import math
from typing import List, Tuple, Optional

import depthai as dai
from depthai_nodes import ImgDetectionsExtended
from depthai_nodes.utils import AnnotationHelper

# ----- same layout helper you use in GridLayoutNode -----
def _layout_rects(n: int) -> List[Tuple[float, float, float, float]]:
    R: List[Tuple[float, float, float, float]] = []
    def row(cols: int, y0: float, h: float, count: Optional[int] = None):
        if count is None: count = cols
        w = 1.0 / cols
        for i in range(count):
            R.append((i * w, y0, w, h))

    if n <= 1:
        return [(0,0,1,1)]
    if n == 2:
        row(2,0,1,2)
    elif n == 3:
        row(2,0,0.5,2); R.append((0,0.5,1,0.5))
    elif n == 4:
        row(2,0,0.5,2); row(2,0.5,0.5,2)
    elif n == 5:
        row(3,0,0.5,3); row(2,0.5,0.5,2)
    elif n == 6:
        row(3,0,0.5,3); row(3,0.5,0.5,3)
    else:
        rows = math.ceil(n/3); h = 1.0/rows; left=n; y=0.0
        for _ in range(rows):
            c = min(3,left); row(c,y,h,c); left -= c; y += h
    return R


class CropGazeAnnotationNode(dai.node.HostNode):
    """
    Input: GatherData.out where msg.gathered is List[ImgDetectionsExtended] from Stage-2 NN,
           aligned to the same face order used by the crop mosaic.
    Output: annotation overlay sized to the mosaic (W,H) that draws eye squares per tile.
    """
    def __init__(self) -> None:
        super().__init__()
        self._in = self.createInput()
        self.out = self.createOutput()

        self._mosaic_w = 2560
        self._mosaic_h = 1440
        self._crop_w = 2560   # dimensions of the crop frames used to build the mosaic tiles
        self._crop_h = 1440
        self.EYE_SCALE = 0.25 # proportional to face size, same as your AnnotationNode

    def build(
        self,
        stage2_gather_out: dai.Node.Output,
        mosaic_size: Tuple[int, int],
        crop_size: Tuple[int, int],
    ) -> "CropGazeAnnotationNode":
        self._mosaic_w, self._mosaic_h = map(int, mosaic_size)
        self._crop_w, self._crop_h = map(int, crop_size)
        self.link_args(stage2_gather_out)
        return self

    # ------- helpers -------

    def _letterbox_params(self, W: int, H: int) -> Tuple[float, float, float]:
        """
        Returns (s, pad_x, pad_y) used when fitting a crop of size (crop_w, crop_h)
        into a tile of size (W,H) with letterbox. Must match GridLayoutNode.
        """
        iw, ih = self._crop_w, self._crop_h
        s = min(W / iw, H / ih)
        pad_x = (W - s * iw) * 0.5
        pad_y = (H - s * ih) * 0.5
        return s, pad_x, pad_y

    def _draw_eye_squares_on_mosaic_tile(
        self,
        ann: AnnotationHelper,
        tile_rect_pix: Tuple[int, int, int, int],  # (X, Y, W, H) in mosaic pixels
        det: "ImgDetectionExtended",                # Stage-2 detection for this crop
    ) -> None:
        """
        Map crop-normalized keypoints into the tile area (with identical letterbox
        math as the mosaic), then draw proportional squares.
        """
        X, Y, W, H = tile_rect_pix
        s, pad_x, pad_y = self._letterbox_params(W, H)

        # face size in the crop (normalized), convert to pixels after scale 's'
        face_w_px = det.rotated_rect.size.width  * self._crop_w * s
        face_h_px = det.rotated_rect.size.height * self._crop_h * s
        side_px = max(6, int(min(face_w_px, face_h_px) * self.EYE_SCALE))

        # helper to convert tile-pixel coords -> mosaic-normalized [0..1]
        def to_norm(px: float, py: float) -> Tuple[float, float]:
            return px / self._mosaic_w, py / self._mosaic_h

        # YuNet: keypoints[0]=left eye, [1]=right eye
        for kp in det.keypoints[:2]:
            # kp.x/y are normalized within the crop
            cx = X + pad_x + kp.x * (self._crop_w * s)
            cy = Y + pad_y + kp.y * (self._crop_h * s)

            half = side_px * 0.5
            x0, y0 = to_norm(cx - half, cy - half)
            x1, y1 = to_norm(cx + half, cy + half)
            ann.draw_rectangle([x0, y0], [x1, y1])

    # ------- main process -------

    def process(self, bundle) -> None:
        """
        'bundle' is a GatherData message:
           - bundle.reference_data == ImgDetectionsExtended from stage-1 (not used here)
           - bundle.gathered       == List[ImgDetectionsExtended] from stage-2 (one per face crop)
        """
        stage2_list = getattr(bundle, "gathered", []) or []
        n = len(stage2_list)

        # Build identical grid layout as your mosaic
        layout = _layout_rects(n)

        ann = AnnotationHelper()

        for idx, det_msg in enumerate(stage2_list):
            if not isinstance(det_msg, ImgDetectionsExtended):
                continue
            if not det_msg.detections:
                continue

            # Tile rectangle in mosaic pixel coordinates
            x, y, w, h = layout[idx]
            W = max(1, int(w * self._mosaic_w))
            H = max(1, int(h * self._mosaic_h))
            X = int(x * self._mosaic_w)
            Y = int(y * self._mosaic_h)

            # Use the first detection on that crop (YuNet gives one face per crop)
            det = det_msg.detections[0]
            self._draw_eye_squares_on_mosaic_tile(ann, (X, Y, W, H), det)

        # Timestamp/seq come from the reference bundle; either works for overlay timing
        ts = None
        seq = None
        try:
            # prefer stage-1 reference (smoother cadence)
            ts = bundle.reference_data.getTimestamp()
            seq = int(bundle.reference_data.getSequenceNum())
        except Exception:
            try:
                ts = stage2_list[-1].getTimestamp()
                seq = int(stage2_list[-1].getSequenceNum())
            except Exception:
                pass

        out = ann.build(timestamp=ts, sequence_num=seq)
        self.out.send(out)
