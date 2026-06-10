"""
Hand Detection + Finger-Drawing Canvas
---------------------------------------
Uses the MediaPipe Tasks API (mediapipe >= 0.10) because the old
mp.solutions API was removed in that version.

Two windows are shown:
  • "Hand Detection"  – webcam feed with skeleton landmarks overlay
  • "Drawing Canvas"  – white board drawn on by your index finger tip

Requirements:
    pip install opencv-python mediapipe numpy

Model file (auto-downloaded already):
    hand_landmarker.task  (must be in the same folder as this script)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

# ──────────────────────────────────────────────
# 1. Locate the model file
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at:\n  {MODEL_PATH}\n"
        "Download it with:\n"
        "  curl -sSL -o hand_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )

# ──────────────────────────────────────────────
# 2. Build the HandLandmarker (Tasks API)
# ──────────────────────────────────────────────
# BaseOptions tells MediaPipe where the model weights live.
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

# HandLandmarkerOptions configures detection behaviour:
#   running_mode  = VIDEO → optimised for continuous frame-by-frame processing
#   num_hands     = 1     → we only need one hand for drawing
#   min_hand_detection_confidence / min_tracking_confidence → same as before
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,   # frame-by-frame video mode
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)
print("✅  HandLandmarker initialised (MediaPipe Tasks API)")

# ──────────────────────────────────────────────
# 3. Open the webcam
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌  Could not open webcam. Grant camera access in:")
    print("    System Settings → Privacy & Security → Camera")
    raise SystemExit(1)

print("✅  Webcam opened.  Press 'q' to quit  |  'c' to clear canvas.")

# ──────────────────────────────────────────────
# 4. Create the blank white drawing canvas
# ──────────────────────────────────────────────
# Read one frame just to get the resolution.
ret, sample = cap.read()
if not ret:
    print("❌  Could not read an initial frame.")
    raise SystemExit(1)

frame_h, frame_w = sample.shape[:2]

# np.ones(shape, dtype) * 255  →  white image (BGR 255,255,255)
# dtype=np.uint8 matches OpenCV's expected 0-255 range per channel
canvas = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255

# ──────────────────────────────────────────────────────────────────────
# WHY LANDMARK 8?
# The hand model returns 21 landmarks (0–20):
#   0  = Wrist
#   4  = Thumb tip
#   8  = Index finger tip  ← most natural "pointer"
#   12 = Middle finger tip
#   16 = Ring finger tip
#   20 = Pinky tip
#
# Landmark 8 sits at the very tip of the index finger, making it the
# most intuitive choice for a drawing cursor.
# ──────────────────────────────────────────────────────────────────────
INDEX_TIP_ID = 8       # landmark index for index finger tip
draw_color   = (220, 80, 20)   # Default BGR → deep blue-ish
draw_radius  = 5               # Default pixels
is_drawing   = True            # Toggle for drawing mode

# ──────────────────────────────────────────────
# 5. Hand-skeleton drawing helper
# ──────────────────────────────────────────────
# Connections between the 21 landmark indices (same as mp.solutions days)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # Thumb
    (0,5),(5,6),(6,7),(7,8),           # Index
    (5,9),(9,10),(10,11),(11,12),      # Middle
    (9,13),(13,14),(14,15),(15,16),    # Ring
    (13,17),(17,18),(18,19),(19,20),   # Pinky
    (0,17),                            # Palm base
]

def draw_skeleton(image, landmarks, frame_w, frame_h):
    """Draw landmark dots and bone connections on `image`."""
    # Convert normalised landmarks to pixel coords for this frame
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]

    # Draw connections (bones)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, pts[start], pts[end], (80, 180, 80), 2, cv2.LINE_AA)

    # Draw landmark dots
    for i, (px, py) in enumerate(pts):
        color = (0, 0, 220) if i == INDEX_TIP_ID else (255, 100, 50)
        radius = 6 if i == INDEX_TIP_ID else 4
        cv2.circle(image, (px, py), radius, color, -1, cv2.LINE_AA)

# ──────────────────────────────────────────────
# 6. Main loop
# ──────────────────────────────────────────────
timestamp_ms = 0   # monotonically increasing timestamp for VIDEO mode

# Variable to track previous frame's index finger tip position
# We initialize this outside the loop.
prev_px, prev_py = None, None
position_history = []   # Stores the last 5 coordinates for smoothing

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Failed to grab frame. Exiting.")
        break

    # Mirror for natural feel
    frame = cv2.flip(frame, 1)

    # ── Convert to MediaPipe Image (RGB) ──────────────────────────────
    # MediaPipe Tasks API wraps images in mp.Image instead of raw arrays.
    # IMAGE_FORMAT.SRGB expects an RGB byte array.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ── Run detection ──────────────────────────────────────────────────
    # VIDEO mode requires a monotonically increasing timestamp (ms).
    timestamp_ms += 33   # ~30 fps
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # ── Process results ────────────────────────────────────────────────
    
    # Optional: Clear a small rectangle on the blank white canvas so our confidence score doesn't smear
    cv2.rectangle(canvas, (frame_w - 230, 0), (frame_w, 40), (255, 255, 255), -1)
    
    if detection_result.hand_landmarks:
        for idx in range(len(detection_result.hand_landmarks)):
            hand_lms = detection_result.hand_landmarks[idx]
            
            # MediaPipe Tasks API returns confidence scores in 'handedness'
            confidence = detection_result.handedness[idx][0].score
            
            # Display the real-time confidence score on both windows
            score_text = f"Confidence: {confidence:.2f}"
            cv2.putText(canvas, score_text, (frame_w - 220, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, score_text, (frame_w - 220, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
            
            # ── 0. Confidence Check ────────────────────────────────────
            # Why a confidence threshold? The model can sometimes falsely identify 
            # background shapes or blurry movement as a hand. A strict threshold (> 0.7) 
            # ensures we only track and draw when we're highly certain it's a real hand.
            if confidence < 0.7:
                continue

            # Draw skeleton on webcam feed
            draw_skeleton(frame, hand_lms, frame_w, frame_h)

            # ── 1. Loop through ALL landmarks and convert them ─────────
            # Landmark guide:
            #   0 = Wrist
            #   4 = Thumb tip
            #   8 = Index finger tip
            #  12 = Middle finger tip
            #  16 = Ring finger tip
            #  20 = Pinky tip
            all_pixel_coords = []
            for i, lm in enumerate(hand_lms):
                # Extra the normalized x, y coordinates (0 to 1)
                norm_x = lm.x
                norm_y = lm.y
                
                # Convert normalized coordinates to pixel coordinates
                pixel_x = int(norm_x * frame_w)
                pixel_y = int(norm_y * frame_h)
                all_pixel_coords.append((pixel_x, pixel_y))
                
                # Print real-time coordinates for the index finger tip (Landmark 8)
                if i == 8:
                    # Using '\033[K\r' nicely updates the print on the same console line without spamming
                    print(f"Index Tip (8) | Norm: {norm_x:.3f}, {norm_y:.3f} | Pixel: {pixel_x}, {pixel_y} \033[K", end="\r")

            # ── 2. Get index finger tip for drawing ────────────────────
            # all_pixel_coords[8] has the raw pixel coordinates for our index tip
            raw_px, raw_py = all_pixel_coords[8]

            # ── 2b. Add smoothing (Average last 5 points) ──────────────
            # Why smooth? Hand tracking models can jitter slightly frame to frame due to noise, 
            # lighting, or tiny movements. By averaging the last 5 positions, we smooth out 
            # these micro-jitters, creating a much cleaner, natural-looking cursor movement.
            position_history.append((raw_px, raw_py))
            if len(position_history) > 5:
                position_history.pop(0)    # remove the oldest position to keep only the last 5
            
            # Calculate the average (mean) of the history list
            avg_x = sum(p[0] for p in position_history) / len(position_history)
            avg_y = sum(p[1] for p in position_history) / len(position_history)
            px, py = int(avg_x), int(avg_y)

            # ── 3. Draw on the canvas (Continuous Line) ────────────────
            # Why store the previous position?
            # Cameras capture at 30-60 frames per second. If you move your finger fast,
            # drawing individual circles leaves a disjointed trail of dots. 
            # Connecting the previous position to the current position ensures a solid, continuous line.
            
            if is_drawing:
                if prev_px is not None and prev_py is not None:
                    # ── 4. Distance / Jump Check ───────────────────────
                    # We calculate the straight-line distance between the current and previous pixel point.
                    # A distance > 50 pixels between 2 frames (~33ms apart) implies an impossibly 
                    # fast, teleporting movement, which is almost certainly a tracking glitch.
                    # We only draw the line segment if distance is reasonable (< 50).
                    distance = ((px - prev_px)**2 + (py - prev_py)**2) ** 0.5
                    
                    if distance < 50:
                        # cv2.line() connects two points with a straight line.
                        cv2.line(canvas, (prev_px, prev_py), (px, py), draw_color, thickness=draw_radius * 2)
                    else:
                        # Massive tracking glitch occurred - reset history so the line is cleanly broken
                        position_history.clear()
                        px, py = raw_px, raw_py   # start fresh without dragging the glitch into the average
                else:
                    # If we have no previous position (like when the hand first enters the frame),
                    # just draw a single dot to start the stroke
                    cv2.circle(canvas, (px, py), draw_radius, draw_color, thickness=-1)
                
            # Update the previous position to the current one for next frame's connective line
            prev_px, prev_py = px, py

            # Live cursor overlay on webcam feed (always visible)
            cv2.circle(frame, (px, py), draw_radius + 4, (0, 255, 0), 2)   # green ring
            cv2.circle(frame, (px, py), 2, draw_color, -1)                 # display brush color inside cursor
            
    else:
        # Reset the previous position and history if no hands are detected in this frame.
        # This prevents the script from drawing a straight line connecting old coordinates
        # to wherever the hand happens to re-enter the camera frame next time.
        prev_px, prev_py = None, None
        position_history.clear()

    # ── HUD on webcam feed ─────────────────────────────────────────────
    hand_count = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
    cv2.putText(frame, f"Hands: {hand_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "q=quit  c=clear  s=save  d=draw mode", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Colors: r, g, b   Size: +, -", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    mode_text = "Drawing: ON" if is_drawing else "Drawing: OFF"
    cv2.putText(frame, f"{mode_text} (Index finger)", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 180), 1, cv2.LINE_AA)

    # ── Show both windows ──────────────────────────────────────────────
    cv2.imshow("Hand Detection \u2014 MediaPipe", frame)
    cv2.imshow("Drawing Canvas", canvas)

    # ── Keyboard controls ──────────────────────────────────────────────
    # cv2.waitKey(1) waits 1 ms for a key event. Returns the ASCII code of the key pressed.
    # We use bitwise AND (& 0xFF) to ensure we get just the lowest 8 bits.
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        print("\n👋  Quitting.")
        break
    elif key == ord("c"):
        # Clear the canvas by filling it with white (BGR 255, 255, 255)
        canvas[:] = 255
        print("\n🗑️   Canvas cleared.")
    elif key == ord("r"):
        # Set drawing color to Red (BGR: 0, 0, 255 - OpenCV uses BGR order, not RGB)
        draw_color = (0, 0, 255)
        print("\n🔴  Color changed to Red.")
    elif key == ord("g"):
        # Set drawing color to Green (BGR: 0, 255, 0)
        draw_color = (0, 255, 0)
        print("\n🟢  Color changed to Green.")
    elif key == ord("b"):
        # Set drawing color to Blue (BGR: 255, 0, 0)
        draw_color = (255, 0, 0)
        print("\n🔵  Color changed to Blue.")
    elif key == ord("s"):
        # Save the current canvas to an image file
        cv2.imwrite("drawing.png", canvas)
        print("\n💾  Canvas saved as 'drawing.png'.")
    elif key == ord("d"):
        # Toggle the drawing mode on or off
        is_drawing = not is_drawing
        state_str = "ON" if is_drawing else "OFF"
        print(f"\n✏️   Drawing mode: {state_str}")
        # Reset previous tracking point so we don't draw a harsh line when re-enabling
        prev_px, prev_py = None, None
    elif key == ord("+") or key == ord("="):
        # Increase brush size
        draw_radius = min(50, draw_radius + 2)
        print(f"\n➕  Brush size increased to {draw_radius}.")
    elif key == ord("-") or key == ord("_"):
        # Decrease brush size
        draw_radius = max(1, draw_radius - 2)
        print(f"\n➖  Brush size decreased to {draw_radius}.")

# ──────────────────────────────────────────────
# 7. Clean up
# ──────────────────────────────────────────────
landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("✅  Resources released. Goodbye!")
