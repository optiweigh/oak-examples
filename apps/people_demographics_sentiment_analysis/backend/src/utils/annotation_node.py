from typing import List
import depthai as dai

from depthai_nodes.utils import AnnotationHelper
from depthai_nodes import Predictions, Classifications

RED = (1.0, 0.0, 0.0, 1.0)
RED_TRANSPARENT = (1.0, 0.0, 0.0, 0.1)
GREEN = (0.0, 1.0, 0.0, 1.0)
GREEN_TRANSPARENT = (0.0, 1.0, 0.0, 0.1)
BLUE = (0.0, 0.0, 1.0, 1.0)
BLUE_TRANSPARENT = (0.0, 0.0, 1.0, 0.1)
WHITE = (1.0, 1.0, 1.0, 1.0)
GRAY = (0.7, 0.7, 0.7, 1.0)
GRAY_TRANSPARENT = (0.7, 0.7, 0.7, 0.1)
TEXT_BG = (0.0, 0.0, 0.0, 0.35)

LIVE = (dai.Tracklet.TrackingStatus.TRACKED, dai.Tracklet.TrackingStatus.NEW)


class AnnotatePeopleFaces(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()

    def build(self, joined: dai.Node.Output) -> "AnnotatePeopleFaces":
        self.link_args(joined)
        return self

    def process(self, msg: dai.Buffer) -> None:
        face_groups: List[dai.MessageGroup] = msg.gathered
        tracklets: dai.Tracklets = msg.reference_data

        ann = AnnotationHelper()

        # draw one label per tracklet; if face attrs missing, show only track ID
        for tl, face in zip(tracklets.tracklets, face_groups):
            if tl.status in LIVE:
                x1, y1 = tl.roi.topLeft().x, tl.roi.topLeft().y
                x2, y2 = tl.roi.bottomRight().x, tl.roi.bottomRight().y

                label = f"ID {tl.id}"

                if not self.mg_is_empty(face):

                    msg_names = set(face.getMessageNames())
                    rid_status = self._get_text(face, "rid_status") if "rid_status" in msg_names else None

                    # Color by status
                    if rid_status == "TBD":
                        color = RED
                        color_fill = RED_TRANSPARENT
                    elif rid_status == "NEW":
                        rid = self._get_text(face, "re_id")
                        color = BLUE
                        color_fill = BLUE_TRANSPARENT
                        label = f"reID: {rid}"
                        label += " | new person"
                    elif rid_status == "REID":
                        rid = self._get_text(face, "re_id")
                        color = GREEN
                        color_fill = GREEN_TRANSPARENT
                        label = f"reID: {rid}"
                        label += " | re-identified"
                    else:
                        color = GRAY

                    label += f"\ntrack ID: {tl.id}"

                    age_msg: Predictions = face["age"]
                    gender_msg: Classifications = face["gender"]
                    emotion_msg: Classifications = face["emotion"]

                    age_pred = getattr(age_msg, "prediction", None)
                    gender = getattr(gender_msg, "top_class", None)
                    if isinstance(age_pred, (int, float)) and gender is not None:
                        age = int(age_pred * 100)
                        if emotion_msg is not None:
                            emotion = emotion_msg.top_class
                        label += f"\n{gender}, {age} | {emotion}"

                else:
                    # No face attrs matched to this tracklet → keep it TBD/RED
                    label = f"track ID: {tl.id}"
                    color = GRAY
                    color_fill = GRAY_TRANSPARENT

                ann.draw_rectangle((x1, y1), (x2, y2), outline_color=color, fill_color=color_fill)
                ann.draw_text(label, (x1 + 0.005, y2 - 0.025), size=18, color=WHITE, background_color=TEXT_BG)

        out = ann.build(timestamp=tracklets.getTimestamp(), sequence_num=tracklets.getSequenceNum())
        self.out.send(out)

    @staticmethod
    def mg_is_empty(mg: dai.MessageGroup) -> bool:
        return mg.getNumMessages() == 0

    @staticmethod
    def _get_text(group: dai.MessageGroup, key: str):
        msg = group[key]
        if isinstance(msg, dai.Buffer):
            return bytes(msg.getData()).decode("utf-8")
        return None
