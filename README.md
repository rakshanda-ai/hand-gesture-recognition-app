# Hand Gesture Recognition App

A modular Python desktop application for real-time hand gesture recognition using MediaPipe, OpenCV, Pillow, and Tkinter.

## Features

- Detects one or two hands from a webcam.
- Draws MediaPipe hand landmarks in real time.
- Recognizes predefined static gestures:
  - Fist
  - Open Palm
  - Pointing
  - Peace
  - Thumbs Up
  - Call Me
  - OK
  - Pinky
  - Four
- Displays camera feed, hand count, handedness confidence, finger states, and gesture names in a modern Tkinter UI.
- Supports both legacy `mp.solutions.hands` and newer `mp.tasks.vision.HandLandmarker` MediaPipe APIs.
- Includes defensive error handling for camera, MediaPipe setup, and frame-processing failures.

## Setup

From the project root:

```powershell
.\.venv-1\Scripts\activate
pip install -r requirements_hand_gesture.txt
python -m hand_gesture_app.tools.download_hand_model
```

The model download is required for newer MediaPipe wheels that expose the Tasks API instead of `mp.solutions`.

## Run

Preferred launcher:

```powershell
python run_hand_gesture_app.py
```

Direct package entry also works:

```powershell
python hand_gesture_app\main.py
```

Make sure your webcam is connected and no other application is using it.

## Project Structure

```text
hand_gesture_app/
  camera.py                     Webcam capture wrapper
  config.py                     Runtime configuration
  detector.py                   MediaPipe hand detection and landmark drawing
  gestures.py                   Rule-based gesture recognition
  gui.py                        Tkinter interface and real-time update loop
  main.py                       Package entry point
  models/hand_landmarker.task   MediaPipe Tasks model file
  tools/download_hand_model.py  Model downloader utility
```

## Notes

Gesture recognition is rule-based and optimized for clear, front-facing hand poses. Lighting, camera angle, and partial occlusion can affect accuracy. For production deployments with larger gesture vocabularies, replace or augment `GestureRecognizer` with a trained classifier while keeping the same module boundary.
