"""
feedback.py — Feedback Manager

Centralizes all user-facing feedback:
  - On-screen text overlay (via OpenCV)
  - Text-to-speech audio (via pyttsx3)

Prevents TTS from repeating the same message on every frame,
and handles the step progress indicator.
"""

import cv2
import pyttsx3

# ─────────────────────────────────────────────
# DISPLAY CONSTANTS
# ─────────────────────────────────────────────

# Main feedback text
FEEDBACK_FONT       = cv2.FONT_HERSHEY_SIMPLEX
FEEDBACK_FONT_SCALE = 1.2
FEEDBACK_COLOR      = (0, 255, 255)   # yellow
FEEDBACK_THICKNESS  = 4
FEEDBACK_POSITION   = (30, 80)        # pixels from top-left

# Step progress indicator (e.g. "Step 2 of 4")
STEP_FONT_SCALE  = 0.8
STEP_COLOR       = (200, 200, 200)    # light gray
STEP_THICKNESS   = 2
STEP_POSITION    = (30, 40)

# "All clear" color override when everything passes
READY_COLOR = (0, 255, 0)   # green


class FeedbackManager:
    """
    Manages TTS and on-screen text feedback.

    Usage:
        fm = FeedbackManager()
        fm.update(frame, feedback_text, current_step, total_steps)
    """

    def __init__(self):
        # TODO: Initialize pyttsx3 TTS engine
        # TODO: Set speech rate (e.g. 160 wpm)
        # TODO: Set voice to macOS Alex: 'com.apple.speech.synthesis.voice.Alex'
        #       (fall back gracefully if not on macOS)
        # TODO: Store last spoken message to avoid repeating
        self._last_spoken = ""

    def speak(self, message):
        """
        Speaks a message via TTS only if it differs from the last spoken message.

        Args:
            message: str
        """

        # TODO: If message == self._last_spoken → do nothing (skip)
        # TODO: Otherwise: engine.say(message), engine.runAndWait()
        # TODO: Update self._last_spoken = message

        pass

    def draw(self, frame, feedback_text, current_step, total_steps):
        """
        Draws the feedback text and step progress onto the frame in-place.

        Args:
            frame: OpenCV BGR image (numpy array) — modified in-place
            feedback_text: str — the main feedback message to display
            current_step: int — 0-indexed current step number
            total_steps: int — total number of steps (excluding READY)
        """

        # TODO: Draw step progress text at STEP_POSITION
        #       Format: "Step {current_step + 1} of {total_steps}"
        #       Use STEP_FONT_SCALE, STEP_COLOR, STEP_THICKNESS
        #       If current_step >= total_steps (READY state), show "All checks passed!"

        # TODO: Choose text color:
        #       If feedback_text indicates all-clear (current_step == READY step) → READY_COLOR
        #       Otherwise → FEEDBACK_COLOR

        # TODO: Draw feedback_text at FEEDBACK_POSITION
        #       Use FEEDBACK_FONT, chosen color, FEEDBACK_FONT_SCALE, FEEDBACK_THICKNESS

        pass

    def update(self, frame, feedback_text, current_step, total_steps):
        """
        Convenience method: draws feedback on frame AND speaks it.

        Args:
            frame: OpenCV BGR image
            feedback_text: str
            current_step: int
            total_steps: int
        """

        # TODO: Call self.draw(frame, feedback_text, current_step, total_steps)
        # TODO: Call self.speak(feedback_text)

        pass
