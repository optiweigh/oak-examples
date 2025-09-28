import depthai as dai


class UnletterboxDetectionsNode(dai.node.HostNode):
    """
    Inputs:
      det_in  : ImgDetections from ParsingNeuralNetwork (bboxes normalized to model canvas Mw×Mh, with letterbox)
      preview : ImgFrame (original frame; only used to get W,H)

    Output:
      out     : ImgDetections with bboxes remapped to ORIGINAL frame (normalized 0..1)

    Build args:
      model_size: (Mw, Mh) model canvas size (e.g., (512, 512))
    """

    def __init__(self):
        super().__init__()
        # define I/O explicitly
        self._in_det = self.createInput()
        self._in_img = self.createInput()
        self._out = self.createOutput()
        self._Mw = 0
        self._Mh = 0

    def build(self, det_in: dai.Node.Output, preview: dai.Node.Output, model_size: tuple):
        self._Mw, self._Mh = int(model_size[0]), int(model_size[1])
        # link order must match process(det_msg, frame)
        self.link_args(det_in, preview)
        return self

    @staticmethod
    def _clamp01(v: float) -> float:
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def _ux(self, xn: float, W: float, s: float, padX: float) -> float:
        # model-normalized -> original-normalized (x)
        x_px = xn * self._Mw
        return self._clamp01(((x_px - padX) / s) / W)

    def _uy(self, yn: float, H: float, s: float, padY: float) -> float:
        # model-normalized -> original-normalized (y)
        y_px = yn * self._Mh
        return self._clamp01(((y_px - padY) / s) / H)

    # IMPORTANT: two-arg signature matches link_args(det_in, preview)
    def process(self, det_msg, frame):
        # frame gives original W,H per timestamp
        if not (hasattr(frame, "getWidth") and hasattr(frame, "getHeight")):
            return
        W = float(frame.getWidth())
        H = float(frame.getHeight())

        # letterbox params used for Mw×Mh canvas
        s = min(self._Mw / W, self._Mh / H)
        padX = (self._Mw - W * s) * 0.5
        padY = (self._Mh - H * s) * 0.5

        out = dai.ImgDetections()

        dets = getattr(det_msg, "detections", []) or []
        new_dets = []
        for d in dets:
            new = dai.ImgDetection()
            if d.label >= 0:
                new.label = d.label
            new.confidence = d.confidence

            if not (hasattr(d, "xmin") and hasattr(d, "ymin") and hasattr(d, "xmax") and hasattr(d, "ymax")):
                continue
            x0 = self._ux(float(d.xmin), W, s, padX)
            y0 = self._uy(float(d.ymin), H, s, padY)
            x1 = self._ux(float(d.xmax), W, s, padX)
            y1 = self._uy(float(d.ymax), H, s, padY)
            if x1 <= x0 or y1 <= y0:
                x0 = y0 = x1 = y1 = 0.0

            new.xmin, new.ymin, new.xmax, new.ymax = x0, y0, x1, y1
            new_dets.append(new)

        out.detections = new_dets
        out.setTimestamp(det_msg.getTimestamp())
        out.setSequenceNum(det_msg.getSequenceNum())
        out.setTimestampDevice(det_msg.getTimestampDevice())
        transformation = det_msg.getTransformation()
        if transformation is not None:
            out.setTransformation(transformation)

        self.out.send(out)
