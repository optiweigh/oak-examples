from typing import List, Tuple, Optional
from dataclasses import dataclass
import depthai as dai
import numpy as np
from depthai_nodes import ImgDetectionsExtended, Classifications, Predictions, GatheredData


@dataclass
class FaceFeature:
    bbox: Tuple[float, float, float, float]
    age: Predictions
    gender: Classifications
    emotion: Classifications
    crop: dai.ImgFrame
    embedding: Optional[np.ndarray]
    rid: Optional[dai.Buffer] = None                # set later by PeopleFacesJoin, not here


class FaceFeaturesMerger:
    """
    A class that matches age, gender, emotion and reid attributes for each face detection.
      - age_gender_gd (GatheredData)
      - emotions_gd (GatheredData)
      - crops_gd (GatheredData)
      - reid_gd (GatheredData)
    Returns:
      - faces: List[FaceFeature]
    """
    def merge(self, age_gender: GatheredData, emotions: GatheredData, crops: GatheredData, reid: GatheredData) -> List[FaceFeature]:
        reference = age_gender.reference_data
        assert isinstance(reference, ImgDetectionsExtended), "Expected ImgDetectionsExtended"

        age_gender_groups = age_gender.gathered
        emotion_groups = emotions.gathered
        reid_groups = reid.gathered
        crop_frames = crops.gathered
        detections = list(reference.detections)

        assert all(isinstance(msg, dai.NNData) for msg in reid_groups), "Expected dai.NNData"

        faces: List[FaceFeature] = []

        for detection, age_gender_msg, emotion_msg, crop_frame, reid_msg in zip(detections, age_gender_groups, emotion_groups, crop_frames, reid_groups):
            bbox = detection.rotated_rect.getOuterRect()
            age_msg: Predictions = age_gender_msg["0"]
            gender_msg: Classifications = age_gender_msg["1"]

            embedding = self._extract_embedding(reid_msg)
            if embedding is not None:
                embedding = self._l2_normalize(embedding)

            face = FaceFeature(
                bbox=bbox,
                age=age_msg,
                gender=gender_msg,
                emotion=emotion_msg,
                crop=crop_frame,
                embedding=embedding,
                rid=None,
            )
            faces.append(face)

        return faces

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
