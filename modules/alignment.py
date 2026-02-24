"""
alignment.py — Step 0: Camera Alignment

Ensures the user is correctly positioned in frame before any clinical
checks begin. This runs first in the guided flow.

Returns: (passed: bool, feedback: str)
"""

import mediapipe as mp

mp_pose = mp.solutions.pose

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

# Shoulder pixel width range that indicates correct distance from camera
MIN_SHOULDER_WIDTH_PX = 200   # too far if below this
MAX_SHOULDER_WIDTH_PX = 410   # too close if above this

# Horizontal center range (as fraction of frame width) for user to be centered
CENTER_MIN = 0.30
CENTER_MAX = 0.70

# Minimum landmark visibility to be considered detected
MIN_VISIBILITY = 0.6


def check_camera_alignment(landmarks, frame_width, frame_height):
    """
    Checks that the user is:
      - Close enough to the camera (shoulder width in pixels)
      - Not too close
      - Horizontally centered in the frame
      - Both shoulders are visible

    Args:
        landmarks: MediaPipe pose landmark list
        frame_width: int, pixel width of the camera frame
        frame_height: int, pixel height of the camera frame

    Returns:
        (passed: bool, feedback: str)
    """

    # TODO: Extract LEFT_SHOULDER, RIGHT_SHOULDER landmarks

    # TODO: Check visibility of both shoulders
    #       If either < MIN_VISIBILITY → return (False, "Move to show both shoulders clearly.")

    # TODO: Calculate pixel distance between shoulders (shoulder_width)
    #       shoulder_width = abs((right_shoulder.x - left_shoulder.x) * frame_width)

    # TODO: Check if too far → return (False, "Move closer to the camera.")

    # TODO: Check if too close → return (False, "Move farther from the camera.")

    # TODO: Calculate horizontal center of both shoulders
    #       center_x = (left_shoulder.x + right_shoulder.x) / 2

    # TODO: Check if too far left → return (False, "Move to the right.")

    # TODO: Check if too far right → return (False, "Move to the left.")

    # TODO: Return (True, "Camera alignment looks good.")

    pass
