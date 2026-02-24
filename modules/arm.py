"""
arm.py — Step 2: Arm Position Check

Checks that the user's arm is correctly positioned for a BP reading.
The arm must be:
  - Resting (supported on a surface, not held up in the air)
  - At roughly heart level (elbow approximately level with the heart)
  - Slightly bent — not fully extended or fully folded

Incorrect arm position can shift BP readings by 10+ mmHg.

Returns: (passed: bool, feedback: str)
"""

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

# Acceptable elbow angle range (degrees) — arm should be slightly bent
# Fully straight arm = ~180°, fully bent = ~30°
# AHA recommends arm supported with elbow slightly bent
MIN_ELBOW_ANGLE = 140   # more bent than this → arm too folded
MAX_ELBOW_ANGLE = 175   # straighter than this → arm too stiff / fully extended

# Heart level is approximated as the midpoint Y between shoulders
# Acceptable range for elbow Y relative to heart Y (as fraction of frame height)
# Positive = elbow is BELOW heart level (too low)
# Negative = elbow is ABOVE heart level (too high)
MAX_ELBOW_BELOW_HEART = 0.10   # elbow can be slightly below heart
MAX_ELBOW_ABOVE_HEART = 0.10   # elbow can be slightly above heart

# Minimum landmark visibility
MIN_VISIBILITY = 0.6


def _angle_between(a, b, c):
    """
    Calculates the angle at point B formed by points A-B-C.

    Args:
        a, b, c: numpy arrays of shape (2,) representing (x, y) pixel coords
                 a = proximal point (shoulder)
                 b = vertex (elbow)
                 c = distal point (wrist)

    Returns:
        angle in degrees (float)
    """

    # TODO: Compute vectors BA and BC
    # TODO: Use np.arccos(dot(BA, BC) / (norm(BA) * norm(BC)))
    # TODO: Convert radians to degrees and return
    # TODO: Handle edge case where norm is 0 (return 0.0)

    pass


def check_arm(landmarks, frame_width, frame_height):
    """
    Checks that the user's left arm is:
      - At approximately heart level (elbow Y ≈ shoulder midpoint Y)
      - Slightly bent at the elbow (angle within MIN_ELBOW_ANGLE–MAX_ELBOW_ANGLE)

    Note: Currently checks left arm only. Right arm support can be added later
    by duplicating logic for RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST.

    Args:
        landmarks: MediaPipe pose landmark list
        frame_width: int
        frame_height: int

    Returns:
        (passed: bool, feedback: str)
    """

    # TODO: Extract LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST landmarks
    #       Also extract RIGHT_SHOULDER to compute heart level

    # TODO: Check visibility of shoulder, elbow, wrist
    #       If any < MIN_VISIBILITY → return (False, "Make sure your left arm is visible.")

    # ── Heart level check ─────────────────────────
    # TODO: Compute heart level Y as average of left and right shoulder Y
    #       heart_y = (left_shoulder.y + right_shoulder.y) / 2
    #
    # TODO: Get elbow Y position (normalized)
    #       elbow_y = left_elbow.y
    #
    # TODO: Compute offset: offset = elbow_y - heart_y
    #       In MediaPipe, Y increases downward
    #       Positive offset → elbow is BELOW heart
    #       Negative offset → elbow is ABOVE heart
    #
    # TODO: If offset > MAX_ELBOW_BELOW_HEART
    #       → return (False, "Raise your arm — your elbow should be at heart level.")
    #
    # TODO: If offset < -MAX_ELBOW_ABOVE_HEART
    #       → return (False, "Lower your arm — your elbow should be at heart level.")

    # ── Elbow angle check ─────────────────────────
    # TODO: Convert shoulder, elbow, wrist to pixel coordinates
    #       (multiply normalized x/y by frame_width/frame_height)
    #
    # TODO: Call _angle_between(shoulder_px, elbow_px, wrist_px) to get elbow angle
    #
    # TODO: If angle < MIN_ELBOW_ANGLE
    #       → return (False, "Straighten your arm slightly.")
    #
    # TODO: If angle > MAX_ELBOW_ANGLE
    #       → return (False, "Bend your elbow slightly and rest your arm on a surface.")

    # TODO: Return (True, "Arm position looks good.")

    pass
