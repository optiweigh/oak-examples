import depthai as dai
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np
import time

from depthai_nodes import GatheredData
from .face_features import FaceFeaturesMerger, FaceFeature
from .math_helpers import bbox_area, bboxes_overlap_area, cos_similarity, norm

LIVE = (dai.Tracklet.TrackingStatus.TRACKED, dai.Tracklet.TrackingStatus.LOST, dai.Tracklet.TrackingStatus.NEW)
REID_MATCH_THRESHOLD = 0.38          # similarity threshold for reidentification of a person
ADAPT_EVERY_N_EMBEDDINGS = 10


@dataclass
class TrackState:
    embeddings: deque
    state: str                       # "TBD" | "NEW" | "REID"
    rid: Optional[str]
    decided: bool


@dataclass
class MemoryEntry:
    embeddings_mean: np.ndarray             # normalized mean embedding
    last_seen: float
    num_samples: int
    adapt_counter: int = 0


class PeopleFacesJoin(dai.node.HostNode):

    """
    A node for matching face detections with person detections (tracklets).
    Internally uses FaceFeaturesMerger to produce face features list.
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

        self.in_track = self.createInput()
        self.in_age_gender = self.createInput()
        self.in_emotions = self.createInput()
        self.in_crops = self.createInput()
        self.in_reid = self.createInput()

        self._face_features_merger = FaceFeaturesMerger()

        self.k_face_samples = 5
        self.tracklet_reid_states: Dict[int, TrackState] = {}

        self.memory: Dict[str, MemoryEntry] = {}
        self.max_memory = 100
        self.next_rid = 0

    def build(self, track: dai.Tracklets, age_gender: dai.Node.Output, emotions: dai.Node.Output, crops: dai.Node.Output, reid: dai.Node.Output):
        self.link_args(track, age_gender, emotions, crops, reid)
        return self

    def process(self,
                track: dai.Tracklets,
                age_gender: dai.Buffer,
                emotions: dai.Buffer,
                crops: dai.Buffer,
                reid: dai.Buffer,
                ) -> None:

        # Merge face attributes (age, gender, emotion, crop, embedding) into a list of FaceFeature objects
        faces: List[FaceFeature] = self._face_features_merger.merge(age_gender=age_gender, emotions=emotions, crops=crops, reid=reid)

        # Filter person tracklets to only include those which are currently LIVE
        people: List[dai.Tracklet] = [tracklet for tracklet in track.tracklets if tracklet.status in LIVE]
        live_tracklets_ids = {tracklet.id for tracklet in people}
        self._remove_terminated_tracklets_states(live_tracklet_ids=live_tracklets_ids)

        aligned_face_features: List[dai.MessageGroup] = [dai.MessageGroup() for _ in people]

        # Match each detected face to a person tracklet based on maximum bounding box overlap
        for face in faces:
            idx, ratio = self._best_person_for_face(face.bbox, people)
            if idx is None or ratio < 0.6:
                continue

            tracklet = people[idx]
            tracklet_state = self._get_reid_state(tracklet_id=tracklet.id)

            # Update RE-ID status
            self._update_reid_state(tracklet_state=tracklet_state, embedding=face.embedding)

            # Build per-person message group
            aligned_face_features[idx] = self._build_msg_group(face=face, tracklet_state=tracklet_state)

        filtered = dai.Tracklets()
        filtered.tracklets = people
        filtered.setTimestamp(track.getTimestamp())
        filtered.setSequenceNum(track.getSequenceNum())

        out = GatheredData(reference_data=filtered, gathered=aligned_face_features)
        out.setTimestamp(track.getTimestamp())
        out.setSequenceNum(track.getSequenceNum())
        self.out.send(out)

    # --------------- Face-Person alignment helpers ---------------

    def _best_person_for_face(self, face_box: Tuple, people: List[dai.Tracklet]) -> Tuple[Optional[int], float]:
        best_idx, best_overlap = None, 0.0
        for i, tracklet in enumerate(people):
            person_box = self._tracklet_to_bbox(tracklet=tracklet)
            overlap = bboxes_overlap_area(person_box, face_box)
            if overlap > best_overlap:
                best_overlap, best_idx = overlap, i

        face_area = bbox_area(face_box)
        ratio = best_overlap / face_area if face_area > 0 else 0.0
        return best_idx, ratio

    @staticmethod
    def _tracklet_to_bbox(tracklet: dai.Tracklet) -> Tuple[float, float, float, float]:
        roi = tracklet.roi
        return (roi.topLeft().x, roi.topLeft().y, roi.bottomRight().x, roi.bottomRight().y)

    # --------------- Per-Tracklet RE-ID state helpers ---------------

    def _get_reid_state(self, tracklet_id: int) -> TrackState:
        state = self.tracklet_reid_states.get(tracklet_id)
        if state is None:
            state = TrackState(
                embeddings=deque(maxlen=self.k_face_samples),
                state="TBD",
                rid=None,
                decided=False
            )
            self.tracklet_reid_states[tracklet_id] = state
        return state

    def _remove_terminated_tracklets_states(self, live_tracklet_ids: set[int]) -> None:
        """
        Removes track state entries for person tracklets that are no longer live/active.
        """
        remove = [k for k in self.tracklet_reid_states.keys() if k not in live_tracklet_ids]
        for k in remove:
            self.tracklet_reid_states.pop(k, None)

    def _update_reid_state(self, tracklet_state: "TrackState", embedding: Optional[np.ndarray]) -> None:
        """
        Ingest one embedding sample if applicable. If enough samples are collected,
        finalize the REID decision and set state/rid/decided on tracklet_state.
        """
        if embedding is None:
            return

        if tracklet_state.decided and tracklet_state.rid is not None:
            self._adapt_memory(tracklet_state.rid, embedding)
            return

        tracklet_state.embeddings.append(embedding)
        if len(tracklet_state.embeddings) >= self.k_face_samples:
            embeddings_mean = self._embeddings_mean(list(tracklet_state.embeddings))
            match_rid = self._match_memory(embeddings_mean)

            if match_rid is None:
                rid = self._promote_new(embeddings_mean=embeddings_mean, num_samples=len(tracklet_state.embeddings))
                tracklet_state.state, tracklet_state.rid, tracklet_state.decided = "NEW", rid, True
            else:
                tracklet_state.state, tracklet_state.rid, tracklet_state.decided = "REID", match_rid, True

    def _embeddings_mean(self, embeddings: List[np.ndarray]) -> np.ndarray:
        return norm(np.mean(embeddings, axis=0))

    # --------------- RE-ID Memory helpers ---------------

    def _match_memory(self, embeddings_mean: np.ndarray) -> Optional[str]:
        """
        Compares embedding with embeddings in memory. If a person is reidentified returns best matching RID.
        """
        if not self.memory:
            return None
        best_rid, best_similarity = None, -1.0
        for rid, entry in self.memory.items():
            similarity = cos_similarity(embeddings_mean, entry.embeddings_mean)
            if similarity > best_similarity:
                best_rid, best_similarity = rid, similarity
        if best_rid is not None and best_similarity >= REID_MATCH_THRESHOLD:
            # Mark as recently used
            self._mark_used(best_rid)
            return best_rid
        return None

    def _adapt_memory(self, rid: str, new_embedding: np.ndarray) -> None:
        """
        Refine the stored embedding for a known RID.
        """
        entry = self.memory.get(rid)
        if entry is None:
            return

        entry.adapt_counter += 1

        if entry.adapt_counter % ADAPT_EVERY_N_EMBEDDINGS != 0:
            return

        similarity = cos_similarity(new_embedding, entry.embeddings_mean)
        if similarity < 0.5:
            return

        n = min(entry.num_samples, 100)
        updated_mean = (entry.embeddings_mean * n + new_embedding) / (n + 1)
        entry.embeddings_mean = norm(updated_mean)
        entry.num_samples = min(n + 1, 100)
        entry.last_seen = time.time()
        entry.adapt_counter %= ADAPT_EVERY_N_EMBEDDINGS

    def _promote_new(self, embeddings_mean: np.ndarray, num_samples: int) -> str:
        """
        Insert a new identity into memory and return its RID.
        Trims memory to self.max_memory keeping most-recently-used first.
        """
        rid = str(self.next_rid)
        self.next_rid += 1
        self.memory[rid] = MemoryEntry(
            embeddings_mean=embeddings_mean,
            last_seen=time.time(),
            num_samples=num_samples,
        )
        self.trim_memory(self.max_memory)
        return rid

    def _mark_used(self, rid: str) -> None:
        """
        Mark an RID as recently seen (for trimming).
        """
        self.memory[rid].last_seen = time.time()

    def trim_memory(self, keep: int) -> None:
        """
        Trim memory, keeping most recently used.
        """
        if len(self.memory) <= keep:
            return
        # Newest first
        items = sorted(self.memory.items(), key=lambda kv: kv[1].last_seen, reverse=True)[:keep]
        self.memory = dict(items)

    # --------------- Payload helpers ---------------

    def _build_msg_group(self, face: FaceFeature, tracklet_state: "TrackState") -> dai.MessageGroup:
        mg = dai.MessageGroup()
        mg["age"] = face.age
        mg["gender"] = face.gender
        mg["emotion"] = face.emotion
        mg["crop"] = face.crop

        if tracklet_state.state:
            mg["rid_status"] = self._string_to_dai_buffer(tracklet_state.state)
        if tracklet_state.rid:
            mg["re_id"] = self._string_to_dai_buffer(tracklet_state.rid)

        return mg

    def _string_to_dai_buffer(self, s: str) -> dai.Buffer:
        buf = dai.Buffer()
        buf.setData(s.encode("utf-8"))
        return buf
