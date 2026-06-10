"""
Hand Landmark Skeleton Visualizer
---------------------------------
This script detects hands and draws ALL 21 landmarks.
Instead of just drawing the index finger, this visualises the full skeleton:
 - Points 0-20 are drawn as color-coded circles.
 - Numbers 0-20 are written out beside each corresponding joint.
 - Lines are drawn connecting the joints to form the hand's "skeleton."

Requirements:
    pip install opencv-python mediapipe numpy

Model file (must be present in folder):
    hand_landmarker.task
"""

import os
import cv2
import time
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ──────────────────────────────────────────────
# Locate the model file
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# ──────────────────────────────────────────────
# Build the HandLandmarker (Tasks API)
# ──────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                       # Detect up to 2 hands to visualise
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = mp_vision.HandLandmarker.create_from_options(options)

print("✅  Visualizer initialised. Opening webcam...")

# ──────────────────────────────────────────────
# Skeleton Configuration
# ──────────────────────────────────────────────
# BGR colors for each finger's joints
COLOR_WRIST  = (0, 0, 255)       # Red    (Wrist)
COLOR_THUMB  = (255, 0, 0)       # Blue   (Thumb)
COLOR_INDEX  = (0, 255, 0)       # Green  (Index)
COLOR_MIDDLE = (0, 255, 255)     # Yellow (Middle)
COLOR_RING   = (0, 165, 255)     # Orange (Ring)
COLOR_PINKY  = (255, 0, 128)     # Purple (Pinky)

def get_color_for_landmark(idx):
    """Returns the color rule for a specific joint index."""
    if idx == 0:
        return COLOR_WRIST
    elif 1 <= idx <= 4:
        return COLOR_THUMB
    elif 5 <= idx <= 8:
        return COLOR_INDEX
    elif 9 <= idx <= 12:
        return COLOR_MIDDLE
    elif 13 <= idx <= 16:
        return COLOR_RING
    elif 17 <= idx <= 20:
        return COLOR_PINKY
    return (255, 255, 255)

# The bone segments to draw lines between
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb branch
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger branch
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger branch
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger branch
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky branch
    (0, 17)                                   # Connect wrist base to palm outer side
]

# ──────────────────────────────────────────────
# Angle Calculation Math
# ──────────────────────────────────────────────
def calculate_angle(start, joint, end):
    """
    Why use vectors?
    Vectors give us direction and magnitude. By capturing the two "bones" meeting
    at a joint as two vectors radiating _outward_ from the joint, we can mathematically 
    measure the exact rotation between them.
    
    The dot product formula states:
    A · B = |A| * |B| * cos(theta)
    
    Rearranging for theta (the angle):
    theta = arccos( (A · B) / (|A| * |B|) )
    
    The angle represents how "bent" the finger is. Approx 180° means the finger 
    is perfectly straight. As you curl the finger, the angle drops towards 0°.
    """
    # Convert points to NumPy coordinate arrays for vector math
    a = np.array(start)  # First point (e.g., lower joint)
    b = np.array(joint)  # Middle point (the joint we are measuring)
    c = np.array(end)    # End point (e.g., upper joint / tip)
    
    # Create vectors originating from the central joint
    ba = a - b
    bc = c - b
    
    # Calculate the dot product and the magnitude of the vectors
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # Calculate cosine of the angle
    # We clip it between -1.0 and 1.0 to prevent floating point precision errors
    # from crashing the arccos function (which only accepts values in that range)
    cosine_angle = np.clip(dot_product / (magnitude_ba * magnitude_bc), -1.0, 1.0)
    
    # Convert inverse cosine (radians) to degrees
    # If points perfectly overlap (e.g. tracking glitch), magnitude is 0 causing NaN
    # We fallback to 0.0 to prevent crashing the script
    angle = np.degrees(np.arccos(cosine_angle))
    return np.nan_to_num(angle, nan=0.0)

# Define which joint indexes map to which fingers to keep our loop clean
FINGER_JOINTS = [
    ("Thumb MCP", 1, 2, 3), ("Thumb IP", 2, 3, 4),
    ("Index PIP", 5, 6, 7), ("Index DIP", 6, 7, 8),
    ("Middle PIP", 9, 10, 11), ("Middle DIP", 10, 11, 12),
    ("Ring PIP", 13, 14, 15), ("Ring DIP", 14, 15, 16),
    ("Pinky PIP", 17, 18, 19), ("Pinky DIP", 18, 19, 20),
]

# ──────────────────────────────────────────────
# Start Webcam
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
timestamp_ms = 0

# CSV Recording State
is_recording = False
csv_file = None
csv_writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Frame dropped. Exiting.")
        break

    # Mirror horizontally for natural view
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    # Convert to MediaPipe task RGB Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect
    timestamp_ms += 33
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Process hands
    if detection_result.hand_landmarks:
        for hand_lms in detection_result.hand_landmarks:
            
            # --- 1. Map all landmarks to Pixel Coordinates ---
            pts = []
            for lm in hand_lms:
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                pts.append((px, py))

            # --- 2. Draw the Skeleton Lines ---
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                # Get the pixel coordinates for the two endpoints of the bone
                pt1 = pts[start_idx]
                pt2 = pts[end_idx]
                # Draw a white line connecting the two points
                cv2.line(frame, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

            # --- 3. Draw the Dots & Text ---
            for i, (px, py) in enumerate(pts):
                # Look up the colour for this specific finger/joint
                color = get_color_for_landmark(i)
                
                # Draw the filled circle (radius 5)
                cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
                # Optional: draw a thin black outline to make colors pop
                cv2.circle(frame, (px, py), 5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Write the landmark number slightly offset from the dot
                # Offset by +8 X and -8 Y so the text sits top-right of the dot
                cv2.putText(
                    frame, 
                    str(i), 
                    (px + 8, py - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4,                  # Font scale/size
                    (255, 255, 255),      # Font colour (White)
                    1,                    # Font thickness
                    cv2.LINE_AA
                )

            # --- 4. Calculate and Display Finger Joint Angles ---
            # We draw this in a list down the right side of the screen
            y_offset = 30
            
            # Keep a dict of current angles so we can easily map them out for the CSV later
            current_angles = {}
            
            # Draw a black semi-transparent background box for the text
            cv2.rectangle(frame, (frame_w - 320, 10), (frame_w - 10, 240), (0, 0, 0), -1)
            
            for name, p1_idx, p2_idx, p3_idx in FINGER_JOINTS:
                # Grab the pixel coordinates for the 3 points forming the joint
                start_pt = pts[p1_idx]
                joint_pt = pts[p2_idx]
                end_pt   = pts[p3_idx]
                
                # Math out the angle
                angle = calculate_angle(start_pt, joint_pt, end_pt)
                current_angles[p2_idx] = (name, angle)
                
                # State Categorization
                if angle < 60:
                    state = "Flexed"
                    text_color = (0, 0, 255)      # Red (BGR)
                elif angle > 120:
                    state = "Extended"
                    text_color = (0, 255, 0)      # Green
                else:
                    state = "Bent"
                    text_color = (0, 255, 255)    # Yellow
                
                # Print the result overlay
                text = f"{name}: {state} ({int(angle)}\u00b0)"
                cv2.putText(frame, text, (frame_w - 310, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
                y_offset += 20

            # --- 5. Log Data to CSV (if recording) ---
            if is_recording and csv_writer is not None:
                current_timestamp = time.time()
                # Create a row for each point (21 total rows per frame)
                for i in range(21):
                    px, py = pts[i]
                    
                    # If this joint has an angle mapped to it, log it, otherwise blank
                    if i in current_angles:
                        joint_name, log_angle = current_angles[i]
                        # Format float to 2 decimal places to stay tidy
                        log_angle = f"{log_angle:.2f}"
                    else:
                        joint_name = f"Landmark_{i}"
                        log_angle = ""
                        
                    csv_writer.writerow([current_timestamp, joint_name, px, py, log_angle])

    # HUD Instructions
    cv2.putText(frame, "Press 'q' to quit  |  'r' to toggle recording", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Blinking red RECORDING indicator
    if is_recording:
        cv2.putText(frame, "RECORDING \u23FA", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Render
    cv2.imshow("Hand Landmarks Visualizer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n👋  Quitting.")
        break
    elif key == ord('r'):
        if not is_recording:
            # Start Recording
            filename = f"hand_data_{int(time.time())}.csv"
            csv_file = open(filename, mode='w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["timestamp", "joint_name", "x_pixel", "y_pixel", "angle"])
            is_recording = True
            print(f"\n🔴 Recording started: Saving to '{filename}'...")
        else:
            # Stop Recording
            is_recording = False
            if csv_file:
                csv_file.close()
            csv_writer = None
            print("\n⏹️  Recording stopped.")

# Optional cleanup to ensure file closes successfully when the loop breaks
if csv_file is not None and not csv_file.closed:
    csv_file.close()

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("✅  Done!")
