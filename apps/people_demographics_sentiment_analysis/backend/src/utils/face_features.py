from typing import List, Tuple, Optional
from dataclasses import dataclass
import depthai as dai
import numpy as np
from depthai_nodes import ImgDetectionsExtended, Classifications, Predictions


@dataclass
class FaceFeature:
    face_det_conf: float
    bbox: Tuple[float, float, float, float]
    age: Predictions
    gender: Classifications
    emotion: Classifications
    crop: dai.ImgFrame
    embedding: Optional[np.ndarray]
    rid: Optional[dai.Buffer] = None                # set later by PeopleFacesJoin, not here


@dataclass
class FaceData:
    sequence_number: int
    faces: List[FaceFeature]


class FaceFeaturesMerger:
    """
    A class that matches age, gender, emotion and reid attributes for each face detection.
      - age_gender_gd (GatheredData)
      - emotions_gd (GatheredData)
      - crops_gd (GatheredData)
      - reid_gd (GatheredData)
    Returns:
      FaceData(sequence_number, list of FaceFeature)
    """
    def merge(self, age_gender_gd, emotions_gd, crops_gd, reid_gd) -> FaceData:
        ref = age_gender_gd.reference_data
        assert isinstance(ref, ImgDetectionsExtended), "Expected ImgDetectionsExtended"

        age_gender_groups = age_gender_gd.gathered
        emotion_groups = emotions_gd.gathered
        reid_groups = reid_gd.gathered
        crop_frames = crops_gd.gathered
        detections = list(ref.detections)

        assert all(isinstance(msg, dai.NNData) for msg in reid_groups)

        n = min(len(detections), len(age_gender_groups), len(emotion_groups), len(crop_frames), len(reid_groups))

        faces: List[FaceFeature] = []

        for i in range(n):
            det = detections[i]
            bbox = det.rotated_rect.getOuterRect()
            age_msg = age_gender_groups[i]["0"]
            gender_msg = age_gender_groups[i]["1"]
            emotion_msg = emotion_groups[i]
            crop_frame = crop_frames[i]

            embedding = self._extract_embedding(reid_groups[i])
            if embedding is not None:
                embedding = self._l2_normalize(embedding)

            face = FaceFeature(
                face_det_conf=det.confidence,
                bbox=bbox,
                age=age_msg,
                gender=gender_msg,
                emotion=emotion_msg,
                crop=crop_frame,
                embedding=embedding,
                rid=None
            )
            faces.append(face)

        return FaceData(
            sequence_number=int(ref.getSequenceNum()),
            faces=faces
        )

    @staticmethod
    def _extract_embedding(nn_msg) -> Optional[np.ndarray]:
        embedding = nn_msg.getTensor("output", dequantize=True)
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return arr

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / (n + 1e-8)
