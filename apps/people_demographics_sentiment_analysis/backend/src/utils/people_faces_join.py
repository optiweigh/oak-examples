from typing import Dict, List, Optional, Tuple
import depthai as dai
from collections import deque
from dataclasses import dataclass
import numpy as np

from depthai_nodes import GatheredData, ImgDetectionsExtended
from .face_features import FaceFeaturesMerger, FaceData, FaceFeature

LIVE = (dai.Tracklet.TrackingStatus.TRACKED, dai.Tracklet.TrackingStatus.NEW)
REID_MATCH_THRESHOLD = 0.4          # similarity threshold for reidentification of a person


@dataclass
class TrackState:
    embeddings: deque
    state: str                       # "TBD" | "NEW" | "REID"
    rid: Optional[str]
    decided: bool


@dataclass
class MemoryEntry:
    embeddings_mean: np.ndarray             # normalized mean embedding


class PeopleFacesJoin(dai.node.ThreadedHostNode):

    """
    A node for matching face detections with person detections (tracklets).
    Internally uses FaceFeaturesMerger to produce FaceData(face features list).
    Inputs:
      - in_tracklets: dai.Tracklets (list of person tracklets with id, status, roi)
      - in_age_gender: GatheredData (reference_data=ImgDetectionsExtended, gathered=[MessageGroup {"0": age(Predictions), "1": gender(Classifications)}])
      - in_emotions: GatheredData (reference_data=ImgDetectionsExtended, gathered=[Classifications])
      - in_crops: GatheredData (reference_data=ImgDetectionsExtended, gathered=[dai.ImgFrame])
      - in_reid: GatheredData (reference_data=ImgDetectionsExtended, gathered=[dai.NNData])
    Output:
      - out: GatheredData (reference_data=filtered dai.Tracklets, gathered=[MessageGroup aligned to each person, empty if no face matched])
    """
    def __init__(self) -> None:
        super().__init__()

        self.in_tracklets = self.createInput()
        self.in_age_gender = self.createInput()
        self.in_emotions = self.createInput()
        self.in_crops = self.createInput()
        self.in_reid = self.createInput()
        self.out = self.createOutput()

        self._tracklets_buffer: Dict[int, dai.Tracklets] = {}
        self._age_gender_buffer: Dict[int, GatheredData] = {}
        self._emotions_buffer: Dict[int, GatheredData] = {}
        self._reid_buffer: Dict[int, GatheredData] = {}
        self._crops_buffer: Dict[int, GatheredData] = {}

        self._face_merger = FaceFeaturesMerger()

        self.k_face_samples = 5
        self.state_by_tl_id: Dict[int, TrackState] = {}
        self.memory: Dict[str, MemoryEntry] = {}
        self.next_rid = 0

    def build(self, tracklets, age_gender, emotions, crops, reid):
        tracklets.link(self.in_tracklets)
        age_gender.link(self.in_age_gender)
        emotions.link(self.in_emotions)
        crops.link(self.in_crops)
        reid.link(self.in_reid)
        return self

    def run(self) -> None:
        while self.isRunning():

            tl = self.in_tracklets.tryGet()
            age_gender_msg = self.in_age_gender.tryGet()
            emotions_msg = self.in_emotions.tryGet()
            reid_msg = self.in_reid.tryGet()
            crop_msg = self.in_crops.tryGet()

            if tl is not None:
                k = self.get_key(tl)
                self._tracklets_buffer[k] = tl

            if age_gender_msg is not None:
                k = self.get_key_gd(age_gender_msg)
                self._age_gender_buffer[k] = age_gender_msg

            if emotions_msg is not None:
                k = self.get_key_gd(emotions_msg)
                self._emotions_buffer[k] = emotions_msg

            if crop_msg is not None:
                k = self.get_key_gd(crop_msg)
                self._crops_buffer[k] = crop_msg

            if reid_msg is not None:
                k = self.get_key_gd(reid_msg)
                self._reid_buffer[k] = reid_msg

            for key in list(self._tracklets_buffer.keys()):
                if (key in self._age_gender_buffer and key in self._emotions_buffer and key in self._crops_buffer and key in self._reid_buffer):
                    self._merge(key)

    def _merge(self, k: int) -> None:

        track = self._tracklets_buffer.pop(k, None)
        age_gender = self._age_gender_buffer.pop(k, None)
        emotion = self._emotions_buffer.pop(k, None)
        reid = self._reid_buffer.pop(k, None)
        crop = self._crops_buffer.pop(k, None)

        if track is None or age_gender is None or emotion is None or crop is None or reid is None:
            return

        face_data: FaceData = self._face_merger.merge(age_gender, emotion, crop, reid)
        faces: List[FaceFeature] = face_data.faces

        # Filter live tracklets
        people: List[dai.Tracklet] = [tl for tl in track.tracklets if tl.status in LIVE]

        live_tl_ids = {tl.id for tl in people}
        self.state_by_tl_id = {k: v for k, v in self.state_by_tl_id.items() if k in live_tl_ids}

        aligned: List[dai.MessageGroup] = [dai.MessageGroup() for _ in people]

        for face in faces:
            face_box = face.bbox

            best_overlap, best_idx = 0.0, -1

            for i, tl in enumerate(people):
                person_box = (
                    tl.roi.topLeft().x, tl.roi.topLeft().y,
                    tl.roi.bottomRight().x, tl.roi.bottomRight().y
                )
                overlap = self._overlap_area(person_box, face_box)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i

            face_area = self._area(face_box)
            if best_idx >= 0 and face_area > 0 and best_overlap >= 0.6 * face_area:

                tl_id = people[best_idx].id
                tl_state: TrackState = self._get_reid_state(tl_id)

                # Add the embedding only while undecided (TBD)
                emb = getattr(face, "embedding", None)
                if emb is not None and not tl_state.decided:
                    tl_state.embeddings.append(emb)

                # Decide when K samples collected
                if (not tl_state.decided) and (len(tl_state.embeddings) >= self.k_face_samples):
                    emb_mean = self._embeddings_mean(list(tl_state.embeddings))
                    match_rid, _ = self._match_memory(emb_mean)
                    if match_rid is None:
                        rid = self._promote_new(emb_mean)
                        tl_state.state, tl_state.rid, tl_state.decided = "NEW", rid, True
                    else:
                        tl_state.state, tl_state.rid, tl_state.decided = "REID", match_rid, True

                mg = dai.MessageGroup()
                mg["age"] = face.age
                mg["gender"] = face.gender
                mg["emotion"] = face.emotion
                mg["crop"] = face.crop

                if tl_state.state:
                    st_buf = dai.Buffer()
                    st_buf.setData(tl_state.state.encode("utf-8"))
                    mg["rid_status"] = st_buf

                if tl_state.rid:
                    rid_buf = dai.Buffer()
                    rid_buf.setData(tl_state.rid.encode("utf-8"))
                    mg["re_id"] = rid_buf

                aligned[best_idx] = mg

        filtered = dai.Tracklets()
        filtered.tracklets = people
        filtered.setTimestamp(track.getTimestamp())
        filtered.setSequenceNum(track.getSequenceNum())

        out = GatheredData(reference_data=filtered, gathered=aligned)
        out.setTimestamp(track.getTimestamp())
        out.setSequenceNum(track.getSequenceNum())
        self.out.send(out)

    def _get_reid_state(self, tl_id: int) -> TrackState:
        state = self.state_by_tl_id.get(tl_id)
        if state is None:
            state = TrackState(
                embeddings=deque(maxlen=self.k_face_samples),
                state="TBD",
                rid=None,
                decided=False
            )
            self.state_by_tl_id[tl_id] = state
        return state

    def _match_memory(self, proto: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.memory:
            return None, -1.0
        best_rid, best_similarity = None, -1.0
        for rid, entry in self.memory.items():
            similarity = self._cos(proto, entry.embeddings_mean)
            # print(similatity, 'rid: ', rid)
            if similarity > best_similarity:
                best_rid, best_similarity = rid, similarity
        # print('best similarity', best_similarity, best_rid)
        return (best_rid if best_similarity >= REID_MATCH_THRESHOLD else None), best_similarity

    def _promote_new(self, proto: np.ndarray) -> str:
        rid = str(self.next_rid)
        self.next_rid += 1
        self.memory[rid] = MemoryEntry(embeddings_mean=proto)
        return rid

    def _embeddings_mean(self, embs: List[np.ndarray]) -> np.ndarray:
        return self._norm(np.mean(embs, axis=0))

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / (n + 1e-8)

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def _area(box):
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _overlap_area(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        if x2 > x1 and y2 > y1:
            return (x2 - x1) * (y2 - y1)
        return 0

    @staticmethod
    def get_key_gd(gd: GatheredData) -> int:
        ref = gd.reference_data
        assert isinstance(ref, ImgDetectionsExtended), f"Expected ImgDetectionsExtended, got {type(ref)}"
        return int(ref.getSequenceNum())

    @staticmethod
    def get_key(msg) -> int:
        if hasattr(msg, "getSequenceNum"):
            return int(msg.getSequenceNum())
        else:
            return int(msg.getTimestamp().total_seconds() * 1000.0)
