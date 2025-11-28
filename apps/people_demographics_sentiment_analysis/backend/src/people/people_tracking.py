from typing import List, Optional
import depthai as dai


class PeopleTrackingStage:
    """
    Builds the people detection + tracking stage of the pipeline.
    """

    def __init__(
        self,
        pipeline: dai.Pipeline,
        image_source: dai.Node.Output,
        fps: int,
        archive: dai.NNArchive,
        labels_to_track: Optional[List[int]] = None,
        det_conf_threshold: float = 0.65,
        tracker_threshold: float = 0.5,
        tracklet_lifespan: int = 15,
        tracker_type: dai.TrackerType = dai.TrackerType.SHORT_TERM_IMAGELESS,
        id_assignment: dai.TrackerIdAssignmentPolicy = dai.TrackerIdAssignmentPolicy.UNIQUE_ID,
    ):
        self._pipeline = pipeline
        self._image_source = image_source
        self._fps = fps

        self._archive = archive
        self._model_w = archive.getInputWidth()
        self._model_h = archive.getInputHeight()
        self._det_conf_threshold = det_conf_threshold

        self._labels_to_track = labels_to_track or [0]  # 0 for person
        self._tracker_threshold = tracker_threshold
        self._tracklet_lifespan = tracklet_lifespan
        self._tracker_type = tracker_type
        self._id_assignment = id_assignment

        self._img_manip: Optional[dai.node.ImageManip] = None
        self._detector: Optional[dai.node.DetectionNetwork] = None
        self._tracker: Optional[dai.node.ObjectTracker] = None

    def build(self) -> "PeopleTrackingStage":
        self._img_manip = self._create_image_manip()

        self._detector = self._create_detector()

        self._tracker = self._create_object_tracker(
            detection_src=self._detector.out, frame_src=self._detector.passthrough
        )

        return self

    def _create_image_manip(self) -> dai.node.ImageManip:
        img_manip = self._pipeline.create(dai.node.ImageManip)
        img_manip.initialConfig.setOutputSize(self._model_w, self._model_h)
        img_manip.initialConfig.setReusePreviousImage(False)
        img_manip.inputImage.setBlocking(True)
        self._image_source.link(img_manip.inputImage)
        return img_manip

    def _create_detector(self) -> dai.node.DetectionNetwork:
        detector = self._pipeline.create(dai.node.DetectionNetwork).build(
            self._img_manip.out,
            self._archive,
            self._fps,
        )
        detector.setConfidenceThreshold(self._det_conf_threshold)
        return detector

    def _create_object_tracker(
        self, detection_src: dai.Node.Output, frame_src: dai.Node.Output
    ) -> dai.node.ObjectTracker:
        tracker = self._pipeline.create(dai.node.ObjectTracker)
        tracker.setDetectionLabelsToTrack(self._labels_to_track)
        tracker.setTrackerThreshold(self._tracker_threshold)
        tracker.setTrackletMaxLifespan(self._tracklet_lifespan)
        tracker.setTrackerType(self._tracker_type)
        tracker.setTrackerIdAssignmentPolicy(self._id_assignment)

        # Linking
        detection_src.link(tracker.inputDetections)
        frame_src.link(tracker.inputTrackerFrame)
        frame_src.link(tracker.inputDetectionFrame)

        return tracker

    @property
    def out(self) -> dai.Node.Output:
        """Tracklets stream"""
        if self._tracker is None:
            raise RuntimeError(
                "PeopleTrackingStage.build() must be called before accessing tracker output."
            )
        return self._tracker.out
