"""Rule-based gesture recognition from MediaPipe hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass


TIP_IDS = (4, 8, 12, 16, 20)
PIP_IDS = (3, 6, 10, 14, 18)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class GestureResult:
    """Classified gesture details for one hand."""

    name: str
    fingers: tuple[bool, bool, bool, bool, bool]

    @property
    def raised_count(self) -> int:
        return sum(self.fingers)


class GestureRecognizer:
    """Recognize common static hand gestures from normalized landmarks."""

    def classify(self, landmarks: object, handedness: str) -> GestureResult:
        """Classify one hand.

        MediaPipe returns normalized coordinates with y increasing downward.
        The thumb is evaluated horizontally while other fingers use fingertip
        position relative to their proximal interphalangeal joints.
        """
        points = landmarks.landmark
        fingers = self._finger_states(points, handedness)
        name = self._gesture_name(fingers, points)
        return GestureResult(name=name, fingers=fingers)

    def _finger_states(self, points: object, handedness: str) -> tuple[bool, bool, bool, bool, bool]:
        thumb_tip = points[4]
        thumb_ip = points[3]
        handedness_lower = handedness.lower()

        if handedness_lower == "right":
            thumb_up = thumb_tip.x < thumb_ip.x
        elif handedness_lower == "left":
            thumb_up = thumb_tip.x > thumb_ip.x
        else:
            thumb_up = abs(thumb_tip.x - thumb_ip.x) > 0.035

        index_up = points[8].y < points[6].y
        middle_up = points[12].y < points[10].y
        ring_up = points[16].y < points[14].y
        pinky_up = points[20].y < points[18].y
        return (thumb_up, index_up, middle_up, ring_up, pinky_up)

    def _gesture_name(self, fingers: tuple[bool, bool, bool, bool, bool], points: object) -> str:
        thumb, index, middle, ring, pinky = fingers

        if fingers == (False, False, False, False, False):
            return "Fist"
        if fingers == (True, True, True, True, True):
            return "Open Palm"
        if fingers == (False, True, False, False, False):
            return "Pointing"
        if fingers == (False, True, True, False, False):
            return "Peace"
        if fingers == (True, False, False, False, False):
            return "Thumbs Up"
        if fingers == (True, False, False, False, True):
            return "Call Me"
        if fingers == (False, False, False, False, True):
            return "Pinky"
        if self._is_ok_sign(points, fingers):
            return "OK"
        if index and middle and ring and pinky and not thumb:
            return "Four"
        return f"{sum(fingers)} Finger{'s' if sum(fingers) != 1 else ''}"

    def _is_ok_sign(self, points: object, fingers: tuple[bool, bool, bool, bool, bool]) -> bool:
        thumb_tip = points[4]
        index_tip = points[8]
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        return distance < 0.055 and fingers[2] and fingers[3] and fingers[4]

