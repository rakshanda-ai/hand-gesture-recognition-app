"""Modern Tkinter GUI for real-time hand gesture recognition."""

from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hand_gesture_app.camera import Camera, CameraError
    from hand_gesture_app.config import APP_CONFIG, AppConfig
    from hand_gesture_app.detector import DetectorError, HandDetector
    from hand_gesture_app.gestures import GestureRecognizer, GestureResult
else:
    from .camera import Camera, CameraError
    from .config import APP_CONFIG, AppConfig
    from .detector import DetectorError, HandDetector
    from .gestures import GestureRecognizer, GestureResult


LOGGER = logging.getLogger(__name__)


class HandGestureApp(tk.Tk):
    """Main desktop application window."""

    def __init__(self, config: AppConfig = APP_CONFIG) -> None:
        super().__init__()
        self.config_model = config
        self.title("Hand Gesture Recognition")
        self.geometry("1180x720")
        self.minsize(960, 620)
        self.configure(bg="#101820")

        self.camera = Camera(config.camera_index, config.frame_width, config.frame_height)
        try:
            self.detector = HandDetector(
                max_hands=config.max_hands,
                detection_confidence=config.detection_confidence,
                tracking_confidence=config.tracking_confidence,
                model_path=config.model_path,
            )
        except DetectorError as exc:
            messagebox.showerror("MediaPipe Setup Error", str(exc))
            raise
        self.recognizer = GestureRecognizer()
        self._running = False
        self._photo: ImageTk.PhotoImage | None = None

        self._build_styles()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#101820")
        style.configure("Panel.TFrame", background="#172430", relief="flat")
        style.configure("Title.TLabel", background="#101820", foreground="#f6f7fb", font=("Segoe UI", 22, "bold"))
        style.configure("Body.TLabel", background="#172430", foreground="#d7dee8", font=("Segoe UI", 11))
        style.configure("Metric.TLabel", background="#172430", foreground="#ffffff", font=("Segoe UI", 20, "bold"))
        style.configure("Muted.TLabel", background="#172430", foreground="#93a4b7", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 8))
        style.map("Accent.TButton", background=[("active", "#00a884")])

    def _build_layout(self) -> None:
        header = ttk.Frame(self, padding=(22, 18, 22, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Hand Gesture Recognition", style="Title.TLabel").pack(side="left")

        main = ttk.Frame(self, padding=(22, 10, 22, 22))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        video_frame = ttk.Frame(main, style="Panel.TFrame", padding=10)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_frame, bg="#0b1117", bd=0)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(main, style="Panel.TFrame", padding=18)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="Status", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(side, text="Camera stopped", style="Metric.TLabel")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(0, 20))

        ttk.Label(side, text="Hands Detected", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        self.hands_count_label = ttk.Label(side, text="0", style="Metric.TLabel")
        self.hands_count_label.grid(row=3, column=0, sticky="w", pady=(0, 20))

        ttk.Label(side, text="Gestures", style="Muted.TLabel").grid(row=4, column=0, sticky="w")
        self.gestures_text = tk.Text(
            side,
            height=8,
            wrap="word",
            bg="#0f1822",
            fg="#f6f7fb",
            insertbackground="#f6f7fb",
            relief="flat",
            padx=12,
            pady=10,
            font=("Segoe UI", 11),
        )
        self.gestures_text.grid(row=5, column=0, sticky="ew", pady=(6, 22))
        self.gestures_text.configure(state="disabled")

        buttons = ttk.Frame(side, style="Panel.TFrame")
        buttons.grid(row=6, column=0, sticky="ew")
        buttons.columnconfigure((0, 1), weight=1)

        self.start_button = ttk.Button(buttons, text="Start", style="Accent.TButton", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="Stop", style="Accent.TButton", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ttk.Label(
            side,
            text="Supported: Fist, Open Palm, Pointing, Peace, Thumbs Up, Call Me, OK, Pinky, Four.",
            style="Body.TLabel",
            wraplength=240,
        ).grid(row=7, column=0, sticky="ew", pady=(24, 0))

    def start(self) -> None:
        """Start webcam capture and recognition."""
        try:
            self.camera.open()
        except CameraError as exc:
            LOGGER.exception("Camera startup failed")
            messagebox.showerror("Camera Error", str(exc))
            return

        self._running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="Live")
        self._update_frame()

    def stop(self) -> None:
        """Stop webcam capture."""
        self._running = False
        self.camera.release()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="Camera stopped")

    def _update_frame(self) -> None:
        if not self._running:
            return

        try:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                raise CameraError("Unable to read a frame from the webcam.")

            frame = cv2.flip(frame, 1)
            hands = self.detector.detect(frame)
            results = [self.recognizer.classify(hand.landmarks, hand.label) for hand in hands]
            self.detector.draw(frame, hands)
            self._draw_labels(frame, hands, results)
            self._render_frame(frame)
            self._update_metrics(hands, results)
        except Exception as exc:
            LOGGER.exception("Frame processing failed")
            self.stop()
            messagebox.showerror("Processing Error", str(exc))
            return

        self.after(self.config_model.update_interval_ms, self._update_frame)

    def _draw_labels(self, frame: cv2.typing.MatLike, hands: list[object], results: list[GestureResult]) -> None:
        for index, (hand, result) in enumerate(zip(hands, results), start=1):
            label = f"{hand.label}: {result.name}"
            y = 38 + (index - 1) * 38
            cv2.rectangle(frame, (16, y - 28), (330, y + 8), (16, 24, 32), -1)
            cv2.putText(frame, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 235, 190), 2, cv2.LINE_AA)

    def _render_frame(self, frame_bgr: cv2.typing.MatLike) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image.thumbnail((self.video_label.winfo_width() or 900, self.video_label.winfo_height() or 560))
        self._photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self._photo)

    def _update_metrics(self, hands: list[object], results: list[GestureResult]) -> None:
        self.hands_count_label.configure(text=str(len(hands)))
        lines: list[str] = []
        for hand, result in zip(hands, results):
            fingers = ", ".join(str(int(state)) for state in result.fingers)
            lines.append(f"{hand.label} ({hand.score:.0%})\nGesture: {result.name}\nFingers: {fingers}")
        self.gestures_text.configure(state="normal")
        self.gestures_text.delete("1.0", "end")
        self.gestures_text.insert("1.0", "\n\n".join(lines) if lines else "No hand detected.")
        self.gestures_text.configure(state="disabled")

    def on_close(self) -> None:
        """Cleanly release resources and close the app."""
        self._running = False
        self.camera.release()
        self.detector.close()
        self.destroy()




