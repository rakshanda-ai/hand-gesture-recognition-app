"""Application entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hand_gesture_app.gui import HandGestureApp
else:
    from .gui import HandGestureApp


def main() -> None:
    """Launch the desktop application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = HandGestureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
