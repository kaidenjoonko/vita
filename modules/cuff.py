"""
cuff.py — Step 3: Cuff Placement Check

Uses the YOLO-detected cuff bounding box and MediaPipe shoulder/elbow
landmarks to determine whether the cuff is placed correctly on the arm.

Correct placement: 2–3 cm above the elbow crease (roughly 55–85% down
the upper arm from shoulder to elbow).

Returns: (passed: bool, feedback: str)
"""

import numpy as np

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────

# Acceptable range for cuff center position along the upper arm
# Expressed as a ratio of (distance from shoulder / total shoulder-to-elbow length)
# 0.0 = at shoulder, 1.0 = at elbow
CUFF_MIN_RATIO = 0.53   # below this → cuff is too high (too close to shoulder)
CUFF_MAX_RATIO = 0.90   # above this → cuff is too low (too close to elbow)

# Minimum landmark visibility
MIN_VISIBILITY = 0.6


def check_cuff_placement(landmarks, cuff_box, frame_width, frame_height):
    """
    Determines whether the detected cuff is positioned correctly on the upper arm.

    Strategy:
      1. Get shoulder and elbow pixel coordinates from MediaPipe
      2. Get the cuff center from the YOLO bounding box
      3. Project the cuff center onto the shoulder→elbow arm vector
      4. Compute position_ratio = projected_length / arm_length
      5. Compare to CUFF_MIN_RATIO and CUFF_MAX_RATIO

    Args:
        landmarks: MediaPipe pose landmark list
        cuff_box: tuple of (x1, y1, x2, y2) pixel coordinates from YOLO detection
                  Pass None if no cuff was detected
        frame_width: int
        frame_height: int

    Returns:
        (passed: bool, feedback: str)
    """

    # TODO: If cuff_box is None → return (False, "No cuff detected. Place the cuff on your upper arm.")

    # TODO: Extract LEFT_SHOULDER and LEFT_ELBOW from landmarks
    #       (right arm support: add RIGHT_SHOULDER, RIGHT_ELBOW later)

    # TODO: Check visibility of both landmarks
    #       If either < MIN_VISIBILITY → return (False, "Arm not visible.")

    # TODO: Convert shoulder and elbow to pixel coordinates
    #       shoulder_px = np.array([left_shoulder.x * frame_width, left_shoulder.y * frame_height])
    #       elbow_px    = np.array([left_elbow.x * frame_width,    left_elbow.y * frame_height])

    # TODO: Compute cuff center from bounding box
    #       x1, y1, x2, y2 = cuff_box
    #       cuff_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # TODO: Compute arm vector and arm length
    #       arm_vec    = elbow_px - shoulder_px
    #       arm_length = np.linalg.norm(arm_vec)
    #       If arm_length == 0 → return (False, "Arm not visible.")

    # TODO: Project cuff center onto arm vector
    #       cuff_vec     = cuff_center - shoulder_px
    #       proj_length  = np.dot(cuff_vec, arm_vec) / arm_length
    #       position_ratio = proj_length / arm_length

    # TODO: If position_ratio < CUFF_MIN_RATIO
    #       → return (False, "Cuff is too high. Move it down, closer to your elbow.")

    # TODO: If position_ratio > CUFF_MAX_RATIO
    #       → return (False, "Cuff is too low. Move it up, about 2 fingers above your elbow.")

    # TODO: Return (True, "Cuff placement looks good.")

    pass
