import os
import cv2
import time
import depthai as dai
from typing import Dict, List, Optional
import numpy as np
from depthai_nodes import GatheredData


class MonitorFacesNode(dai.node.ThreadedHostNode):
    def __init__(self) -> None:
        super().__init__()
        self.in_faces = self.createInput()
        self.out = self.createOutput()

        self.people: Dict[str, Dict] = {}                        # id -> {age, gender, emotion, crop, last_seen}
        self.max_people_slots = 3
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

                    status = None
                    person_id = None

                    if "rid_status" in face.getMessageNames():
                        status = self._get_txt(face["rid_status"])

                    if status == 'TBD':
                        continue

                    if "re_id" in face.getMessageNames():
                        person_id = self._get_txt(face["re_id"])

                    if person_id is None:
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

                    if gender_score < 0.7 or emotion_score < 0.2:
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
                        self.people[person_id] = {"age": age, "gender": gender, "emotion": emotion,
                                                  "crop": face_crop, "last_seen": now, "status": status}
                    else:
                        entry["age"] = age
                        entry["gender"] = gender
                        entry["emotion"] = emotion
                        entry["crop"] = face_crop
                        entry["last_seen"] = now
                        entry["status"] = status

                    if person_id not in seen_ids:
                        seen_ids.append(person_id)

                for person_id in seen_ids:
                    self.move_face_to_front(person_id)

                self.trim_people(keep=7)

            # emit crops + payload at 1Hz
            now2 = time.monotonic()
            if (now2 - self._last_emit_ts) >= self._emit_interval:
                self._emit_seq += 1
                self._last_emit_ts = now2

                # URLs in slot order
                urls_by_slot: List[str] = []
                for s in range(self.max_slots):
                    person_id = self.people_slots[s]
                    if not person_id or person_id not in self.people:
                        urls_by_slot.append(self.placeholder_url())
                        continue
                    crop = self.people[person_id].get("crop")
                    if crop is None:
                        urls_by_slot.append(self.placeholder_url())
                        continue
                    bgr = self.imgframe_to_bgr(crop)
                    if bgr is None or bgr.size == 0:
                        urls_by_slot.append(self.placeholder_url())
                        continue
                    urls_by_slot.append(self.save_slot_jpg(s, bgr))

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

                # payload in slot order (0..2)
                faces_by_slot = []
                for s in range(self.max_slots):
                    person_id = self.people_slots[s]
                    if not person_id or person_id not in self.people:
                        faces_by_slot.append({"img_url": urls_by_slot[s]})
                        continue
                    person = self.people[person_id]
                    faces_by_slot.append({
                        "id": person_id,
                        "status": person.get("status"),
                        "age": person["age"],
                        "gender": person["gender"],
                        "emotion": person["emotion"],
                        "img_url": urls_by_slot[s],
                    })

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
        # remove if already present
        self.people_slots = [p for p in self.people_slots if p != person_id]
        self.people_slots.insert(0, person_id)
        if len(self.people_slots) > self.max_slots:
            self.people_slots = self.people_slots[:self.max_slots]

    def imgframe_to_bgr(self, f: dai.ImgFrame) -> np.ndarray:
        return f.getCvFrame()

    def trim_people(self, keep: int) -> None:
        if len(self.people) <= keep:
            return
        items = sorted(self.people.items(), key=lambda kv: kv[1]["last_seen"], reverse=True)[:keep]
        self.people = dict(items)

    # ---- saving files / urls ----
    def save_slot_jpg(self, slot_idx: int, img_bgr: np.ndarray) -> str:
        fs_path = os.path.join(self.CROPS_DIR, f"slot{slot_idx}.jpg")
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            return f"{self.PLACEHOLDER_URL}?t={self._emit_seq}"
        cv2.imwrite(fs_path, img_bgr)
        return f"{self.CROPS_URL}/slot{slot_idx}.jpg?t={self._emit_seq}"

    def placeholder_url(self) -> str:
        return f"{self.PLACEHOLDER_URL}?t={self._emit_seq}"

    @staticmethod
    def _get_txt(msg: Optional[dai.Buffer]) -> Optional[str]:
        if isinstance(msg, dai.Buffer):
            return bytes(msg.getData()).decode("utf-8")
        return None
