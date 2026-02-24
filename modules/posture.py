"""
posture.py — Step 1: Posture Check

Checks that the user is seated correctly before a blood pressure reading.
Poor posture (slouching, leaning, crossed legs) can raise BP by 6–10 mmHg.

Returns: (passed: bool, feedback: str)
"""

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

# Max allowed difference in Y position between left and right shoulder (as fraction of frame)
# Larger value = more tilt allowed
MAX_SHOULDER_TILT = 0.05

# Min vertical distance between shoulder and hip (as fraction of frame height)
# If too small, user is hunched/slouching
MIN_SHOULDER_HIP_DIST = 0.15

# Max forward lean: how far the shoulder midpoint can be in front of the hip midpoint
# Measured as fraction of frame width
MAX_FORWARD_LEAN = 0.08

# Minimum landmark visibility to be considered detected
MIN_VISIBILITY = 0.6


def check_posture(landmarks, frame_width, frame_height):
    """
    Checks that the user is:
      - Sitting up straight (shoulders not drooping toward hips)
      - Not leaning to one side (shoulder tilt)
      - Not leaning forward
      - Feet flat (legs not crossed) — only checked if knees/ankles are visible

    Args:
        landmarks: MediaPipe pose landmark list
        frame_width: int
        frame_height: int

    Returns:
        (passed: bool, feedback: str)
    """

    # TODO: Extract LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP landmarks

    # TODO: Check visibility of shoulders and hips
    #       If not visible → return (False, "Make sure your upper body and hips are visible.")

    # ── Shoulder tilt ─────────────────────────────
    # TODO: Compute difference in Y between left and right shoulder
    #       (normalized coords — multiply by frame_height for pixels)
    #       If abs(left_shoulder.y - right_shoulder.y) > MAX_SHOULDER_TILT
    #       → return (False, "Sit up straight — you're leaning to one side.")

    # ── Slouching ─────────────────────────────────
    # TODO: Compute vertical distance from shoulder midpoint to hip midpoint
    #       If distance < MIN_SHOULDER_HIP_DIST
    #       → return (False, "Sit up straight — you appear to be slouching.")

    # ── Forward lean ──────────────────────────────
    # TODO: Compare X position of shoulder midpoint vs hip midpoint
    #       MediaPipe x increases left to right
    #       Significant X offset between shoulders and hips indicates leaning forward/backward
    #       If abs(shoulder_mid_x - hip_mid_x) > MAX_FORWARD_LEAN
    #       → return (False, "Sit back — don't lean forward.")

    # ── Legs crossed ──────────────────────────────
    # TODO: Extract LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE
    #       Only run this check if all four landmarks have visibility > MIN_VISIBILITY
    #       (knees/ankles may not be visible if camera is chest-up — skip gracefully)
    #
    #       Legs crossed heuristic:
    #         - Compare X positions of left knee vs right knee
    #         - If left knee X > right knee X (they've swapped sides), legs are likely crossed
    #       → return (False, "Uncross your legs and place both feet flat on the floor.")

    # TODO: Return (True, "Posture looks good.")

    pass
