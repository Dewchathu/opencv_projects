import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open the video file
cap = cv2.VideoCapture('video/sample1.mp4')

# Variables to count push-ups
push_up_count = 0
push_up_phase = 'down'  # 'down' or 'up'
threshold_angle_down = 90  # Threshold angle to detect the 'down' position
threshold_angle_up = 160  # Threshold angle to detect the 'up' position


def calculate_angle(a, b, c):
    """Calculate the angle between three points a, b, c."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the key points
        landmarks = result.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

        # Calculate the angles
        shoulder_angle = calculate_angle(shoulder, elbow, wrist)
        hip_angle = calculate_angle(hip, knee, ankle)

        # Check the phase of the push-up
        if push_up_phase == 'down' and shoulder_angle > threshold_angle_up:
            push_up_phase = 'up'
            push_up_count += 1
        elif push_up_phase == 'up' and shoulder_angle < threshold_angle_down:
            push_up_phase = 'down'

        # Display the push-up count and angles
        cv2.putText(frame, f'Push-ups: {push_up_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Shoulder Angle: {int(shoulder_angle)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        cv2.putText(frame, f'Hip Angle: {int(hip_angle)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Phase: {push_up_phase}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Push-up Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
