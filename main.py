import cv2
import mediapipe as mp
import pyttsx3

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
        if shoulder_width < 100:
            return "Move closer to the camera."
        elif shoulder_width > 400:
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    feedback = "Position yourself in the camera frame."

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        joints = {
            "L Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
            "R Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "L Elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
            "R Elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
            "L Wrist": mp_pose.PoseLandmark.LEFT_WRIST,
            "R Wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
            "L Index": mp_pose.PoseLandmark.LEFT_INDEX,
            "R Index": mp_pose.PoseLandmark.RIGHT_INDEX,
            "L Thumb": mp_pose.PoseLandmark.LEFT_THUMB,
            "R Thumb": mp_pose.PoseLandmark.RIGHT_THUMB,
            "L Hip": mp_pose.PoseLandmark.LEFT_HIP,
            "R Hip": mp_pose.PoseLandmark.RIGHT_HIP,
            "L Knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "R Knee": mp_pose.PoseLandmark.RIGHT_KNEE,
            "L Ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "R Ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
            "Nose": mp_pose.PoseLandmark.NOSE,
            "L Ear": mp_pose.PoseLandmark.LEFT_EAR,
            "R Ear": mp_pose.PoseLandmark.RIGHT_EAR,
            "L Mouth": mp_pose.PoseLandmark.MOUTH_LEFT,
            "R Mouth": mp_pose.PoseLandmark.MOUTH_RIGHT
        }

        for name, idx in joints.items():
            lm = landmarks[idx]
            if lm.visibility > 0.6:
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, name, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        feedback = get_camera_alignment_feedback(landmarks, width, height)
        speak_if_changed(feedback)

    cv2.putText(frame, feedback, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)

    cv2.imshow("Vita - Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
