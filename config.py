"""Application configuration constants."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings for the hand gesture application."""

    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    max_hands: int = 2
    detection_confidence: float = 0.65
    tracking_confidence: float = 0.55
    update_interval_ms: int = 15
    model_path: Path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


APP_CONFIG = AppConfig()
