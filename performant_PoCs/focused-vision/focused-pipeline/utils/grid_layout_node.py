import math
from typing import List, Tuple
import cv2
import numpy as np
import depthai as dai

def _layout_rects(n: int) -> List[Tuple[float, float, float, float]]:
    R: List[Tuple[float, float, float, float]] = []
    def row(cols: int, y0: float, h: float, count: int | None = None):
        if count is None: count = cols
        w = 1.0 / cols
        for i in range(count):
            R.append((i * w, y0, w, h))

    if n <= 1: return [(0,0,1,1)]
    if n == 2: row(2,0,1,2)
    elif n == 3: row(2,0,0.5,2); R.append((0,0.5,1,0.5))
    elif n == 4: row(2,0,0.5,2); row(2,0.5,0.5,2)
    elif n == 5: row(3,0,0.5,3); row(2,0.5,0.5,2)
    elif n == 6: row(3,0,0.5,3); row(3,0.5,0.5,3)
    else:
        rows = math.ceil(n/3); h = 1.0/rows; left=n; y=0.0
        for _ in range(rows):
            c = min(3,left); row(c,y,h,c); left-=c; y+=h
    return R

def _fit_letterbox(img: np.ndarray, W: int, H: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(W/iw, H/ih)
    nw, nh = max(1,int(iw*s)), max(1,int(ih*s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x0 = (W-nw)//2; y0 = (H-nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _mosaic_from_frames(
    frames: List[dai.ImgFrame],
    out_w: int,
    out_h: int,
    frame_type: dai.ImgFrame.Type
) -> dai.ImgFrame:
    # blank if nothing arrived
    if not frames:
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        out = dai.ImgFrame()
        out.setType(frame_type); out.setWidth(out_w); out.setHeight(out_h)
        out.setData(canvas.tobytes())
        return out

    mats: List[np.ndarray] = []
    for fr in frames:
        img = fr.getCvFrame()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        mats.append(img)

    layout = _layout_rects(len(mats))
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for (x, y, w, h), img in zip(layout, mats):
        W = max(1, int(w * out_w))
        H = max(1, int(h * out_h))
        X = int(x * out_w)
        Y = int(y * out_h)
        tile = _fit_letterbox(img, W, H)
        canvas[Y:Y+H, X:X+W] = tile

    out = dai.ImgFrame()
    out.setType(frame_type); out.setWidth(out_w); out.setHeight(out_h)
    out.setData(canvas.tobytes())
    try: out.setTimestamp(frames[-1].getTimestamp())
    except: pass
    try: out.setSequenceNum(int(frames[-1].getSequenceNum()))
    except: pass
    return out

class GridLayoutNode(dai.node.HostNode):
    """
    Input: GatherData.out where msg.gathered is List[dai.ImgFrame] (your crops)
    Output: single mosaic dai.ImgFrame
    """
    def __init__(self):
        super().__init__()
        self.crops_input = self.createInput()
        self.output = self.createOutput()

        self._target_w = 1920
        self._target_h = 1080
        self.frame_type = dai.ImgFrame.Type.BGR888p

    def build(self, crops_input: dai.Node.Output, target_size: Tuple[int, int]) -> "GridLayoutNode":
        self._target_w, self._target_h = map(int, target_size)
        self.link_args(crops_input)
        return self

    def process(self, msg):
        # msg is a GatherData bundle; crops are in msg.gathered (a Python list)
        crops: List[dai.ImgFrame] = getattr(msg, "gathered", [])
        mosaic = _mosaic_from_frames(crops, self._target_w, self._target_h, self.frame_type)
        self.output.send(mosaic)
