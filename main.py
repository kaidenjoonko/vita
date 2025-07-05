import cv2
import mediapipe as mp
import pyttsx3
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model
model = YOLO("runs/detect/train2/weights/best.pt")


# model.fuse()  # Uncomment when for final deployment

# Initialize Text-to-Speech engine with Alex voice
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.Alex')
last_feedback_spoken = ""

def speak_if_changed(message):
    global last_feedback_spoken
    if message != last_feedback_spoken:
        engine.say(message)
        engine.runAndWait()
        last_feedback_spoken = message

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Vita - Pose Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vita - Pose Detection", 1280, 720)
cv2.moveWindow("Vita - Pose Detection", 100, 50)

def get_camera_alignment_feedback(landmarks, image_width, image_height):
    try:
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        if ls.visibility < 0.6 or rs.visibility < 0.6:
            return "Move to show both shoulders clearly."

        shoulder_width = abs((rs.x - ls.x) * image_width)
        if shoulder_width < 200:
            return "Move closer to the camera."
        elif shoulder_width > 410:
            return "Move farther from the camera."

        center_x = (ls.x + rs.x) / 2
        if center_x < 0.3:
            return "Move to the right."
        elif center_x > 0.7:
            return "Move to the left."

        if abs(ls.y - lh.y) < 0.1:
            return "Sit up straight."

        return "Camera alignment looks good."

    except:
        return "Unable to detect landmarks clearly."


def check_cuff_placement(shoulder, elbow, cuff_center):
    arm_vec = elbow - shoulder
    cuff_vec = cuff_center - shoulder
    arm_length = np.linalg.norm(arm_vec)

    if arm_length == 0:
        return "Arm not visible."

    proj_length = np.dot(cuff_vec, arm_vec) / arm_length
    position_ratio = proj_length / arm_length

    if position_ratio < 0.53:
        return "Cuff is too close to shoulder. Lower it."
    elif position_ratio > 0.9:
        return "Cuff is too close to elbow. Raise it."
    else:
        return "Cuff placement looks good."


# ======================= Main Loop =======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Recolor to process
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detections/process
    results = pose.process(img_rgb)

    feedback = "Position yourself in the camera frame."

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        joints = {
            "L Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "L Elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "R Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "R Elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "L Hip": mp_pose.PoseLandmark.LEFT_HIP,
            "R Hip": mp_pose.PoseLandmark.RIGHT_HIP
        }

        for name, idx in joints.items():
            lm = landmarks[idx]
            if lm.visibility > 0.6:
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, name, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Check camera alignment
        alignment_feedback = get_camera_alignment_feedback(landmarks, width, height)
        
        # Check cuff placement if alignment is good
        if alignment_feedback == "Camera alignment looks good.":
            ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]

            if ls.visibility > 0.6 and le.visibility > 0.6:
                shoulder = np.array([ls.x * width, ls.y * height])
                elbow = np.array([le.x * width, le.y * height])

                results_yolo = model.predict(frame, imgsz=640, conf=0.5)
                cuff_found = False

                for box in results_yolo[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cuff_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.circle(frame, (int(cuff_center[0]), int(cuff_center[1])), 6, (0, 0, 255), -1)

                    feedback = check_cuff_placement(shoulder, elbow, cuff_center)
                    cuff_found = True

                if not cuff_found:
                    feedback = "No cuff detected."

            else:
                feedback = "Arm not visible."
        else:
            feedback = alignment_feedback


        speak_if_changed(feedback)

    cv2.putText(frame, feedback, (30, 80),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
    cv2.imshow("Vita - Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
