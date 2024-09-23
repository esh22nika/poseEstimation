'''
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize Mediapipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load your trained model
model = tf.keras.models.load_model('pose_violence_model.h5')

# Define the preprocess_landmarks function
def preprocess_landmarks(landmarks, num_landmarks=33):
    # Initialize an array to hold landmark data
    landmark_array = np.zeros((num_landmarks, 4))

    if landmarks:
        # Populate the array with landmark data
        for i, landmark in enumerate(landmarks):
            if i < num_landmarks:
                landmark_array[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

    # Ensure the array has the expected shape
    return landmark_array.reshape(1, num_landmarks, 4)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize holistic model and face mesh model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame and get the landmarks
        results = holistic.process(image)
        face_results = face_mesh.process(image)

        # Convert the frame back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face mesh landmarks
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

        # Draw holistic landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        # Collect and preprocess landmark data
        pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
        preprocessed_landmarks = preprocess_landmarks(pose_landmarks)

        # Make predictions
        prediction = model.predict(preprocessed_landmarks)
        label = "Violence" if prediction[0][0] > 0.5 else "Non-Violence"  # Adjust according to your model's output

        # Display prediction
        cv2.putText(image, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Raw Webcam Feed', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize Mediapipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load your trained model
model = tf.keras.models.load_model('pose_violence_model.h5')


# Define the preprocess_landmarks function
def preprocess_landmarks(landmarks, num_landmarks=33):
    landmark_array = np.zeros((num_landmarks, 4))
    if landmarks:
        for i, landmark in enumerate(landmarks):
            if i < num_landmarks:
                landmark_array[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
    return landmark_array.reshape(1, num_landmarks, 4)


# Calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Function to detect body-to-body interaction
def detect_body_to_body_interaction(landmarks, distance_threshold=0.1):
    if len(landmarks) < 2:
        return False  # No interaction if there's only one person

    person_1 = landmarks[0]
    person_2 = landmarks[1]

    # Measure distances between key body parts like shoulders and hips
    shoulder_dist = euclidean_distance(
        np.array([person_1[11].x, person_1[11].y, person_1[11].z]),
        np.array([person_2[11].x, person_2[11].y, person_2[11].z])
    )

    hip_dist = euclidean_distance(
        np.array([person_1[23].x, person_1[23].y, person_1[23].z]),
        np.array([person_2[23].x, person_2[23].y, person_2[23].z])
    )

    close_proximity = shoulder_dist < distance_threshold or hip_dist < distance_threshold

    return close_proximity


# Function to assess individual frame interaction features
def analyze_frame_interaction(pose_landmarks, face_landmarks=None):
    interaction_score = 0

    if face_landmarks:
        # Check for proximity between faces for potential face-to-face interaction
        face_dist = euclidean_distance(
            np.array([face_landmarks[0].x, face_landmarks[0].y, face_landmarks[0].z]),
            np.array([face_landmarks[1].x, face_landmarks[1].y, face_landmarks[1].z])
        )
        interaction_score += 1 if face_dist < 0.1 else 0

    # Calculate body-to-body interaction score
    interaction_score += detect_body_to_body_interaction(pose_landmarks)

    return interaction_score


# Initialize video capture
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
        mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the frame and get the landmarks
        results = holistic.process(image)
        face_results = face_mesh.process(image)

        # Convert the frame back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face mesh landmarks
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Draw holistic landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            pose_landmarks = [results.pose_landmarks.landmark]
            preprocessed_landmarks = preprocess_landmarks(pose_landmarks[0])

            # Detect body-to-body interaction and analyze frame interactions
            interaction_detected = detect_body_to_body_interaction(pose_landmarks)
            interaction_score = analyze_frame_interaction(pose_landmarks)

            if interaction_score > 0:
                label = "Violence"
            else:
                label = "Non-Violence" if len(pose_landmarks) < 2 else "Violence"
        else:
            label = "Non-Violence"  # Default to Non-Violence if no landmarks are detected

        # Display prediction
        cv2.putText(image, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Raw Webcam Feed', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.
'''
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import torch
from ultralytics import YOLO
import sys

# Ensure UTF-8 encoding for the terminal output
sys.stdout.reconfigure(encoding='utf-8')

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load YOLO model
yolo_model = YOLO("yolov5s.pt")

# Load your trained violence detection model
violence_model = tf.keras.models.load_model('pose_violence_model.h5')


def preprocess_landmarks(landmarks, num_landmarks=33):
    landmark_array = np.zeros((num_landmarks, 4))
    if landmarks:
        for i, landmark in enumerate(landmarks.landmark):
            if i < num_landmarks:
                landmark_array[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
    return landmark_array.reshape(1, num_landmarks, 4)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def is_hand_touching_face(landmarks, distance_threshold=0.1):
    face_points = [landmarks.landmark[mp_pose.PoseLandmark.NOSE],
                   landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE],
                   landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]]
    hand_points = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
                   landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]]

    for face_point in face_points:
        for hand_point in hand_points:
            dist = euclidean_distance(
                np.array([face_point.x, face_point.y, face_point.z]),
                np.array([hand_point.x, hand_point.y, hand_point.z])
            )
            if dist < distance_threshold:
                return True
    return False


# Initialize video capture
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# For saving the video file as output.avi
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, (frame_width, frame_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # YOLO person detection
        results = yolo_model(frame, classes=[0])  # 0 is the class index for 'person'

        # Process detected persons
        for i, det in enumerate(results[0].boxes.data):
            if det is not None and len(det) == 6:
                x1, y1, x2, y2, conf, cls = det
                if int(cls) == 0:  # Ensure it's a person
                    # Crop the person from the frame
                    person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Convert the cropped image to RGB
                    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                    # Process with MediaPipe Pose
                    results_pose = pose.process(rgb_crop)

                    if results_pose.pose_landmarks:
                        # Draw pose landmarks on the cropped image
                        mp_drawing.draw_landmarks(
                            person_crop,
                            results_pose.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                        )

                        # Preprocess landmarks for violence detection
                        preprocessed_landmarks = preprocess_landmarks(results_pose.pose_landmarks)

                        # Make violence prediction
                        prediction = violence_model.predict(preprocessed_landmarks)
                        model_label = "Violence" if prediction[0][0] > 0.5 else "Non-Violence"

                        # Check for hand-face interaction
                        hand_face_interaction = is_hand_touching_face(results_pose.pose_landmarks)
                        interaction_label = "Violence" if hand_face_interaction else "Non-Violence"

                        # Combine both predictions
                        final_label = "Violence" if model_label == "Violence" or interaction_label == "Violence" else "Non-Violence"

                        # Get nose coordinates
                        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        x = int(nose.x * (x2 - x1) + x1)
                        y = int(nose.y * (y2 - y1) + y1)

                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Display prediction and coordinates
                        cv2.putText(frame, f'Person {i + 1}: {final_label}', (int(x1), int(y1) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f'Coords: ({x}, {y})', (int(x1), int(y1) - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Draw a circle at the nose position
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Write the frame to output video file
        out.write(frame)

        # Display the frame
        cv2.imshow('Multi-person Violence Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()