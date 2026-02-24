import cv2
import mediapipe as mp
from ultralytics import YOLO

from modules.alignment import check_camera_alignment
from modules.posture import check_posture
from modules.arm import check_arm
from modules.cuff import check_cuff_placement
from modules.feedback import FeedbackManager

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Path to the trained YOLO cuff detection model
MODEL_PATH = "runs/detect/train2/weights/best.pt"

# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# MediaPipe detection confidence thresholds
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5

# Guided flow steps — user must pass each in order before advancing
# Each step maps to a check function in modules/
STEPS = [
    "ALIGN_CAMERA",   # Step 0: get user positioned correctly in frame
    "CHECK_POSTURE",  # Step 1: back straight, feet flat, not leaning
    "CHECK_ARM",      # Step 2: arm resting at heart level, correct angle
    "CHECK_CUFF",     # Step 3: cuff detected and placed correctly on arm
    "READY",          # Step 4: all checks passed — user can take reading
]


# ─────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────

# TODO: Load YOLO model from MODEL_PATH

# TODO: Initialize MediaPipe Pose with POSE_DETECTION_CONFIDENCE and POSE_TRACKING_CONFIDENCE

# TODO: Initialize FeedbackManager (handles TTS + on-screen text)

# TODO: Open webcam (cv2.VideoCapture), set resolution to FRAME_WIDTH x FRAME_HEIGHT

# TODO: Create and size the display window ("Vita - BP Assistant")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

# TODO: Track current step index (starts at 0)
# TODO: Track consecutive frames a step has been passing (for stability before advancing)
#       — require N consecutive passing frames before moving to next step
#       — prevents flickering between steps on a single good frame

while True:

    # TODO: Read frame from webcam; break if frame not available

    # TODO: Flip frame horizontally (mirror view feels more natural for users)

    # TODO: Convert BGR → RGB for MediaPipe

    # TODO: Run MediaPipe Pose on the frame

    # ── Step logic ────────────────────────────────

    # TODO: If no pose landmarks detected → feedback = "Stand in front of the camera"
    #       and reset step to 0

    # TODO: Otherwise, run the check for the current step:
    #
    #   Step 0 (ALIGN_CAMERA):
    #       call check_camera_alignment(landmarks, frame_width, frame_height)
    #       → returns (passed: bool, feedback: str)
    #
    #   Step 1 (CHECK_POSTURE):
    #       call check_posture(landmarks, frame_width, frame_height)
    #       → returns (passed: bool, feedback: str)
    #
    #   Step 2 (CHECK_ARM):
    #       call check_arm(landmarks, frame_width, frame_height)
    #       → returns (passed: bool, feedback: str)
    #
    #   Step 3 (CHECK_CUFF):
    #       run YOLO on the frame to get cuff bounding box
    #       call check_cuff_placement(landmarks, cuff_box, frame_width, frame_height)
    #       → returns (passed: bool, feedback: str)
    #
    #   Step 4 (READY):
    #       feedback = "Great! You're ready to take your reading."
    #       no further checks needed

    # TODO: If check passed for N consecutive frames → advance to next step
    #       If check failed → reset consecutive frame counter (do NOT go backwards in steps)

    # ── Rendering ─────────────────────────────────

    # TODO: Draw MediaPipe pose landmarks on frame

    # TODO: Draw YOLO cuff bounding box on frame (only during CHECK_CUFF step)

    # TODO: Draw step progress indicator on frame (e.g. "Step 2 of 4")

    # TODO: Send current feedback to FeedbackManager (handles TTS + on-screen display)

    # TODO: Show frame with cv2.imshow

    # TODO: Break on ESC key (cv2.waitKey)


# ─────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────

# TODO: cap.release()
# TODO: cv2.destroyAllWindows()
