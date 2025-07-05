import cv2
import os

# Create output folder
output_folder = "train"
os.makedirs(output_folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press SPACE to capture a photo.")
print("Press ESC to quit.")

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam - Press SPACE to capture", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = f"cuff_{img_counter:03d}.png"
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, frame)
        print(f"{img_name} saved!")
        img_counter += 1

cap.release()
cv2.destroyAllWindows()