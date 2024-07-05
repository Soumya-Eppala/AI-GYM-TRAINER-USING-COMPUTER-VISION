from flask import Flask, render_template, request
import threading
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# Global variable to store the pop-up message
popup_message = ""

# Function to set the pop-up message
def set_popup_message(message):
    global popup_message
    popup_message = message

def calculate_angle(a, b, c):
    """
    Function to calculate the angle between three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def track_workout(workout_choice):
    global popup_message
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: 
        last_command_time = {}
        workout_ended = False
        rep_count = 0
        last_command = None
        in_down_position = False

        while cap.isOpened() and not workout_ended:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image)

            landmarks = results.pose_landmarks

            if landmarks is not None:
                mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Calculate angles for left and right sides
                if workout_choice == 1:  # Bicep Curls
                    # Left side
                    shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)

                    # Right side
                    shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
                
                elif workout_choice == 2:  # Squats
                    # Left side
                    knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle_left = calculate_angle(hip_left, knee_left, ankle_left)

                    # Right side
                    knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    angle_right = calculate_angle(hip_right, knee_right, ankle_right)

                # Check for incorrect performance
                if angle_right > 160 and angle_left > 160:
                    last_command = "Correct Form"
                    in_down_position = True
                else:
                    if angle_right >= 160:
                        set_popup_message("Your left bicep curl angle is too small. Keep your left arm closer to your body.")
                    elif angle_right <= 30:
                        set_popup_message("Your left bicep curl angle is too large. Keep your left arm at a right angle.")
                    if angle_left >= 160:
                        set_popup_message("Your right bicep curl angle is too small. Keep your right arm closer to your body.")
                    elif angle_left <= 30:
                        set_popup_message("Your right bicep curl angle is too large. Keep your right arm at a right angle.")
                
                # Check for 'e' key press to end workout

                # Your code to display last command and rep count here

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Workout Tracker', image)
                
                # Check if a pop-up message is set and pass it to the template
                if popup_message:
                    with app.app_context():
                        return render_template('index.html', popup_message=popup_message)
            except Exception as e:
                print("Error:", e)
                
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index1.html', popup_message="")

@app.route('/start_workout', methods=['POST'])
def start_workout():
    workout_choice = int(request.form['workout_choice'])

    thread = threading.Thread(target=track_workout, args=(workout_choice,))
    thread.start()

    return ''

if __name__ == '__main__':
    app.run(debug=True)
