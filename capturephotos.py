"""
capturephotos.py — Training Data Collection Utility

Use this script to capture images of the BP cuff for YOLO training.
Images are saved to the train/ folder. After capturing, upload them
to Roboflow (or similar) to draw bounding boxes and export labels.

Controls:
  SPACE  → save current frame as an image
  ESC    → quit

Usage:
  python capturephotos.py
"""

import cv2
import os

OUTPUT_FOLDER = "train"
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Press SPACE to capture a photo.")
print("Press ESC to quit.")

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Show live count on screen
    cv2.putText(frame, f"Saved: {img_counter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE = capture | ESC = quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Capture Training Photos", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print("Closing.")
        break
    elif k == 32:
        img_name = f"cuff_{img_counter:04d}.png"
        img_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()
