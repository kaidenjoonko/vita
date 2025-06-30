import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    results = pose.process(img_rgb)  # Process the image and find pose landmarks

    # Draw landmarks on the original image
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark
        joints = {
            # Sharms
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

            # Torso & Posture
            "L Hip": mp_pose.PoseLandmark.LEFT_HIP,
            "R Hip": mp_pose.PoseLandmark.RIGHT_HIP,

            "L Knee": mp_pose.PoseLandmark.LEFT_KNEE,
            "R Knee": mp_pose.PoseLandmark.RIGHT_KNEE,

            "L Ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
            "R Ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,

            # Head / Face
            "Nose": mp_pose.PoseLandmark.NOSE,
            "L Ear": mp_pose.PoseLandmark.LEFT_EAR,
            "R Ear": mp_pose.PoseLandmark.RIGHT_EAR,
            "L Mouth": mp_pose.PoseLandmark.MOUTH_LEFT,
            "R Mouth": mp_pose.PoseLandmark.MOUTH_RIGHT
        }
        
        for name, idx in joints.items():
            lm = landmarks[idx]
            if lm.visibility > 0.6: # Only draw visible points
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, name, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Vita - Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
