"""Webcam capture helpers with clear error handling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


@dataclass
class Camera:
    """Small wrapper around OpenCV video capture."""

    index: int = 0
    width: int = 960
    height: int = 540

    def __post_init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam and configure its preferred frame size."""
        self._capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = cv2.VideoCapture(self.index)

        if not self._capture.isOpened():
            raise CameraError(
                "Unable to open webcam. Check that a camera is connected and not used by another app."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Read one frame from the webcam."""
        if self._capture is None or not self._capture.isOpened():
            raise CameraError("Camera is not open.")
        return self._capture.read()

    def release(self) -> None:
        """Release webcam resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

