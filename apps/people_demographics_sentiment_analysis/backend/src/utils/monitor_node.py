import os
import cv2
import time
import depthai as dai
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from depthai_nodes import GatheredData

MAX_PEOPLE_SLOTS = 3
KEEP_PEOPLE = 8
GENDER_CONFIDENCE_THR = 0.7
EMOTION_CONFIDENCE_THR = 0.2
ID_DECIDED = {"NEW", "REID"}


@dataclass
class Person:
    age: int
    gender: str
    emotion: str
    crop: dai.ImgFrame
    last_seen: float
    status: str


class MonitorFacesNode(dai.node.ThreadedHostNode):
    """
    Collects face metadata and maintains a small, recency-ordered set of people for FE display.
    Emits a payload with person statistics and per-sloth face info.

    Expected input (from PeopleFacesJoin):
    - in_faces: GatheredData where:
        reference_data: dai.Tracklets
        gathered: List[dai.MessageGroup] aligned to tracklets; each MessageGroup may contain:
        - "rid_status": dai.Buffer -> {"TBD","NEW","REID"}
        - "re_id": dai.Buffer -> string, person id (only when rid_status in {"NEW","REID"})
        - "age": Predictions
        - "gender": Classifications
        - "emotion": Classifications
        - "crop": dai.ImgFrame (face crop)
    """
    def __init__(self) -> None:
        super().__init__()
        self.in_faces = self.createInput()
        self.out = self.createOutput()

        self.people: Dict[str, Person] = {}
        self.max_people_slots = MAX_PEOPLE_SLOTS
        self.people_slots: List[Optional[str]] = [None] * self.max_people_slots

        # Emit at 1 Hz (crops + payload stay in sync)
        self._last_emit_ts = 0.0
        self._emit_interval = 1.0
        self._emit_seq = 0
        self.latest_payload: Optional[dict] = None

        # Stats
        self.ages: list[int] = []
        self.male_cnt = 0
        self.female_cnt = 0
        self.emotion_keys = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        self.emotion_cnt = {k: 0 for k in self.emotion_keys}

        self.CROPS_DIR = "/static-fe/crops"                         # FE path to write
        self.CROPS_URL = "/crops"                                   # URL prefix
        self.PLACEHOLDER_URL = "/placeholders/empty.jpg"
        os.makedirs(self.CROPS_DIR, exist_ok=True)

    def build(self, faces: dai.Node.Output) -> "MonitorFacesNode":
        faces.link(self.in_faces)
        return self

    def run(self) -> None:
        while self.isRunning():
            faces = self.in_faces.tryGet()
            if faces is not None:
                assert isinstance(faces, GatheredData)
                face_groups: List[dai.MessageGroup] = faces.gathered
                now = time.monotonic()

                seen_ids: List[str] = []

                for face in (face_groups):
                    if face.getNumMessages() == 0:
                        continue

                    face_msg_names = set(face.getMessageNames())
                    status = self._get_txt(face["rid_status"]) if "rid_status" in face_msg_names else None
                    if status not in ID_DECIDED:
                        continue

                    person_id = self._get_txt(face["re_id"]) if "re_id" in face_msg_names else None
                    if not person_id:
                        continue

                    age_msg = face["age"]
                    gender_msg = face["gender"]
                    emotion_msg = face["emotion"]
                    face_crop = face["crop"]

                    age_pred = getattr(age_msg, "prediction", None)
                    gender = getattr(gender_msg, "top_class", None)
                    emotion = getattr(emotion_msg, "top_class", None)

                    gender_score = getattr(gender_msg, "top_score", None)
                    emotion_score = getattr(emotion_msg, "top_score", None)

                    if not (isinstance(age_pred, (int, float)) and gender and emotion):
                        continue

                    if gender_score < GENDER_CONFIDENCE_THR or emotion_score < EMOTION_CONFIDENCE_THR:
                        continue

                    age = int(age_pred * 100)

                    self.ages.append(age)
                    if len(self.ages) > 2000:
                        self.ages = self.ages[-1000:]
                    if gender == "Male":
                        self.male_cnt += 1
                    elif gender == "Female":
                        self.female_cnt += 1
                    if emotion in self.emotion_cnt:
                        self.emotion_cnt[emotion] += 1

                    entry = self.people.get(person_id)
                    if entry is None:
                        self.people[person_id] = Person(age=age, gender=gender, emotion=emotion,
                                                        crop=face_crop, last_seen=now, status=status)
                    else:
                        entry.age = age
                        entry.gender = gender
                        entry.emotion = emotion
                        entry.crop = face_crop
                        entry.last_seen = now
                        entry.status = status

                    if person_id not in seen_ids:
                        seen_ids.append(person_id)

                for person_id in seen_ids:
                    self.move_face_to_front(person_id)

                self.trim_people(keep=KEEP_PEOPLE)

            # emit crops + payload at 1Hz
            now2 = time.monotonic()
            if (now2 - self._last_emit_ts) >= self._emit_interval:
                self._emit_seq += 1
                self._last_emit_ts = now2

                faces_by_slot: List[dict] = []

                for idx, person_id in enumerate(self.people_slots):
                    # default: placeholder-only payload
                    url = self.placeholder_url()
                    payload = {"img_url": url}

                    entry = self.people.get(person_id) if person_id else None
                    if entry:
                        crop = entry.crop
                        if crop is not None:
                            bgr = self.imgframe_to_bgr(crop)
                            if bgr is not None and bgr.size != 0:
                                url = self.save_slot_jpg(idx, bgr)
                        payload = {
                            "id": person_id,
                            "status": entry.status,
                            "age": entry.age,
                            "gender": entry.gender,
                            "emotion": entry.emotion,
                            "img_url": url,
                        }

                    faces_by_slot.append(payload)

                # stats
                avg_age = (sum(self.ages) / len(self.ages)) if self.ages else 0.0
                gender_total = max(1, self.male_cnt + self.female_cnt)
                male_percentage = 100.0 * self.male_cnt / gender_total
                female_percentage = 100.0 * self.female_cnt / gender_total
                emotion_total = max(1, sum(self.emotion_cnt.values()))
                emotions_percentage = {k: (100.0 * self.emotion_cnt[k] / emotion_total) for k in self.emotion_keys}
                if gender_total > 1_000_000 or emotion_total > 1_000_000:
                    div = 100
                    self.male_cnt //= div
                    self.female_cnt //= div
                    for k in self.emotion_cnt:
                        self.emotion_cnt[k] //= div

                self.latest_payload = {
                    "seq": self._emit_seq,
                    "timestamp": now2,
                    "faces": faces_by_slot,
                    "stats": {
                        "age": avg_age,
                        "males": male_percentage,
                        "females": female_percentage,
                        "emotions": emotions_percentage
                    }
                }

    def move_face_to_front(self, person_id: str) -> None:
        self.people_slots = [p for p in self.people_slots if p != person_id]
        self.people_slots.insert(0, person_id)
        self._pad_slots()

    def _pad_slots(self) -> None:
        if len(self.people_slots) < self.max_people_slots:
            self.people_slots += [None] * (self.max_people_slots - len(self.people_slots))
        elif len(self.people_slots) > self.max_people_slots:
            self.people_slots = self.people_slots[:self.max_people_slots]

    def imgframe_to_bgr(self, f: dai.ImgFrame) -> np.ndarray:
        return f.getCvFrame()

    def trim_people(self, keep: int) -> None:
        if len(self.people) <= keep:
            return
        items = sorted(self.people.items(), key=lambda kv: kv[1].last_seen, reverse=True)[:keep]
        self.people = dict(items)
        # remove reference from people slots also
        keep_set = set(self.people.keys())
        self.people_slots = [pid if (pid in keep_set) else None for pid in self.people_slots]
        self._pad_slots()

    # ---- saving files / urls ----
    def save_slot_jpg(self, slot_idx: int, img_bgr: np.ndarray) -> str:
        fs_path = os.path.join(self.CROPS_DIR, f"slot{slot_idx}.jpg")
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            return f"{self.PLACEHOLDER_URL}?t={self._emit_seq}"
        cv2.imwrite(fs_path, img_bgr)
        # public URL the FE will request for the face crop
        return f"{self.CROPS_URL}/slot{slot_idx}.jpg?t={self._emit_seq}"

    def placeholder_url(self) -> str:
        return f"{self.PLACEHOLDER_URL}?t={self._emit_seq}"

    @staticmethod
    def _get_txt(msg: Optional[dai.Buffer]) -> Optional[str]:
        if isinstance(msg, dai.Buffer):
            return bytes(msg.getData()).decode("utf-8")
        return None
