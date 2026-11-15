"""MediaPipe hand detection and drawing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import cv2
import mediapipe as mp


class DetectorError(RuntimeError):
    """Raised when MediaPipe cannot be initialized or used."""


@dataclass(frozen=True)
class HandInfo:
    """Detected hand landmarks and handedness metadata."""

    landmarks: object
    label: str
    score: float


class HandDetector:
    """Detect hands and draw landmarks using MediaPipe.

    MediaPipe Python currently appears in two forms: the older
    ``mp.solutions.hands`` API and the newer ``mp.tasks.vision`` API. This
    detector supports both so the rest of the application can remain stable.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.65,
        tracking_confidence: float = 0.55,
        model_path: Path | str | None = None,
    ) -> None:
        self._mode = "tasks"
        self._hands = None
        self._landmarker = None
        self._connections: list[tuple[int, int]] = []

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._init_solutions(max_hands, detection_confidence, tracking_confidence)
        else:
            if model_path is None:
                raise DetectorError("MediaPipe Tasks requires a hand_landmarker.task model file.")
            self._init_tasks(Path(model_path), max_hands, detection_confidence, tracking_confidence)

    def _init_solutions(
        self,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> None:
        self._mode = "solutions"
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._connections = [(start, end) for start, end in self._mp_hands.HAND_CONNECTIONS]
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def _init_tasks(
        self,
        model_path: Path,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> None:
        if not model_path.exists():
            raise DetectorError(
                "MediaPipe hand model not found.\n\n"
                f"Expected: {model_path}\n\n"
                "Run: python -m hand_gesture_app.tools.download_hand_model"
            )

        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._connections = [
            (connection.start, connection.end)
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS
        ]

    def detect(self, frame_bgr: cv2.typing.MatLike) -> list[HandInfo]:
        """Return detected hands for a BGR OpenCV frame."""
        if self._mode == "solutions":
            return self._detect_with_solutions(frame_bgr)
        return self._detect_with_tasks(frame_bgr)

    def _detect_with_solutions(self, frame_bgr: cv2.typing.MatLike) -> list[HandInfo]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        hand_infos: list[HandInfo] = []
        handedness_items = results.multi_handedness or []
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            score = 0.0
            if index < len(handedness_items):
                classification = handedness_items[index].classification[0]
                label = classification.label
                score = classification.score
            hand_infos.append(HandInfo(landmarks=landmarks, label=label, score=score))
        return hand_infos

    def _detect_with_tasks(self, frame_bgr: cv2.typing.MatLike) -> list[HandInfo]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_infos: list[HandInfo] = []
        for index, landmark_list in enumerate(result.hand_landmarks):
            label = "Unknown"
            score = 0.0
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                label = getattr(category, "category_name", None) or getattr(category, "display_name", None) or "Unknown"
                score = float(getattr(category, "score", 0.0))

            hand_infos.append(
                HandInfo(
                    landmarks=SimpleNamespace(landmark=landmark_list),
                    label=label,
                    score=score,
                )
            )
        return hand_infos

    def draw(self, frame_bgr: cv2.typing.MatLike, hands: Iterable[HandInfo]) -> None:
        """Draw hand landmarks in-place on the supplied frame."""
        for hand in hands:
            if self._mode == "solutions":
                self._mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand.landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )
            else:
                self._draw_tasks_landmarks(frame_bgr, hand.landmarks.landmark)

    def _draw_tasks_landmarks(self, frame_bgr: cv2.typing.MatLike, landmarks: list[object]) -> None:
        height, width = frame_bgr.shape[:2]
        points = [(int(point.x * width), int(point.y * height)) for point in landmarks]

        for start, end in self._connections:
            if start < len(points) and end < len(points):
                cv2.line(frame_bgr, points[start], points[end], (0, 190, 255), 2, cv2.LINE_AA)

        for point in points:
            cv2.circle(frame_bgr, point, 5, (0, 235, 190), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, point, 7, (15, 24, 32), 1, cv2.LINE_AA)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()
