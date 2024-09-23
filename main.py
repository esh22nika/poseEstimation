'''
import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Initialize Mediapipe models
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define paths
violence_folder_path = 'small_Violence'
non_violence_folder_path = 'small_nonViolence'
output_folder_path = 'preProcessedData'

# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)


# Process videos in a folder
def process_videos(video_folder_path, label):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for file in os.listdir(video_folder_path):
            output_file_path = os.path.join(output_folder_path, f'{label}_{os.path.splitext(file)[0]}.npy')

            # Skip processing if the output file already exists
            if os.path.exists(output_file_path):
                print(f"Skipping {file}, already preprocessed.")
                continue

            video_path = os.path.join(video_folder_path, file)
            cap = cv2.VideoCapture(video_path)

            # Output list to store all frame data
            output_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess the frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Perform detection
                results = holistic.process(image)

                # Collect the landmarks
                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    output_data.append(np.array(landmarks))  # Append each frame's landmarks as a separate array

                # Visualization (optional)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                cv2.imshow('Frame', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

            # Save preprocessed data
            np.save(output_file_path, output_data)  # Save as a list of arrays
            print(f"Saved: {output_file_path}")

    cv2.destroyAllWindows()


# Process both folders
process_videos(violence_folder_path, 'violence')
process_videos(non_violence_folder_path, 'non_violence')


# Splitting the data into training, validation, and testing sets
def split_data(preprocessed_folder_path):
    violence_files = []
    non_violence_files = []

    # Load all violence and non-violence npy files
    for file in os.listdir(preprocessed_folder_path):
        file_path = os.path.join(preprocessed_folder_path, file)
        if 'violence' in file:
            violence_files.append(file_path)
        elif 'non_violence' in file:
            non_violence_files.append(file_path)

    # Combine and label data
    X = violence_files + non_violence_files
    y = [1] * len(violence_files) + [0] * len(non_violence_files)  # 1 for violence, 0 for non-violence

    # Split data into training + validation and testing sets (e.g., 85% train+val, 15% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    # Further split training + validation set into separate training and validation sets (e.g., 70% train, 15% val)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, stratify=y_train_val,
                                                      random_state=42)

    # Print the sizes of each set to verify
    print(f'Training set size: {len(X_train)}')
    print(f'Validation set size: {len(X_val)}')
    print(f'Testing set size: {len(X_test)}')

    # Example of loading data from the splits
    def load_data(file_paths):
        data = [np.load(file) for file in file_paths]
        return data

    # Load datasets
    train_data = load_data(X_train)
    val_data = load_data(X_val)
    test_data = load_data(X_test)

    # Now you can proceed with training using train_data and validation with val_data
    return train_data, val_data, test_data, X_train, X_val, X_test


# Call the function to split the data
train_data, val_data, test_data, X_train, X_val, X_test = split_data(output_folder_path)
'''

import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Initialize Mediapipe models
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define paths
violence_folder_path = 'small_Violence'
non_violence_folder_path = 'small_nonViolence'
output_folder_path = 'preProcessedData'

# Create output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to detect body-to-body interaction
def detect_body_to_body_interaction(landmarks):
    if len(landmarks) < 2:
        return False  # No interaction if there's only one person

    person_1 = landmarks[0]
    person_2 = landmarks[1]

    shoulder_dist = euclidean_distance(
        np.array([person_1.landmark[11].x, person_1.landmark[11].y, person_1.landmark[11].z]),
        np.array([person_2.landmark[11].x, person_2.landmark[11].y, person_2.landmark[11].z])
    )

    hip_dist = euclidean_distance(
        np.array([person_1.landmark[23].x, person_1.landmark[23].y, person_1.landmark[23].z]),
        np.array([person_2.landmark[23].x, person_2.landmark[23].y, person_2.landmark[23].z])
    )

    close_proximity_threshold = 0.1

    return shoulder_dist < close_proximity_threshold or hip_dist < close_proximity_threshold

# Process videos in a folder
def process_videos(video_folder_path, label):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for file in os.listdir(video_folder_path):
            output_file_path = os.path.join(output_folder_path, f'{label}_{os.path.splitext(file)[0]}.npy')

            # Skip processing if the output file already exists
            if os.path.exists(output_file_path):
                print(f"Skipping {file}, already preprocessed.")
                continue

            video_path = os.path.join(video_folder_path, file)
            cap = cv2.VideoCapture(video_path)

            # Output list to store all frame data
            output_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess the frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Perform detection
                results = holistic.process(image)

                # Collect the landmarks
                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    output_data.append(np.array(landmarks))  # Append each frame's landmarks as a separate array

                # Visualization (optional)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                cv2.imshow('Frame', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

            # Save preprocessed data
            np.save(output_file_path, output_data)  # Save as a list of arrays
            print(f"Saved: {output_file_path}")

    cv2.destroyAllWindows()

# Process both folders
process_videos(violence_folder_path, 'violence')
process_videos(non_violence_folder_path, 'non_violence')

# Splitting the data into training, validation, and testing sets
def split_data(preprocessed_folder_path):
    violence_files = []
    non_violence_files = []

    # Load all violence and non-violence npy files
    for file in os.listdir(preprocessed_folder_path):
        file_path = os.path.join(preprocessed_folder_path, file)
        if 'violence' in file:
            violence_files.append(file_path)
        elif 'non_violence' in file:
            non_violence_files.append(file_path)

    # Combine and label data
    X = violence_files + non_violence_files
    y = [1] * len(violence_files) + [0] * len(non_violence_files)  # 1 for violence, 0 for non-violence

    # Split data into training + validation and testing sets (e.g., 85% train+val, 15% test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    # Further split training + validation set into separate training and validation sets (e.g., 70% train, 15% val)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, stratify=y_train_val,
                                                      random_state=42)

    # Print the sizes of each set to verify
    print(f'Training set size: {len(X_train)}')
    print(f'Validation set size: {len(X_val)}')
    print(f'Testing set size: {len(X_test)}')

    # Example of loading data from the splits
    def load_data(file_paths):
        data = [np.load(file) for file in file_paths]
        return data

    # Load datasets
    train_data = load_data(X_train)
    val_data = load_data(X_val)
    test_data = load_data(X_test)

    # Now you can proceed with training using train_data and validation with val_data
    return train_data, val_data, test_data, X_train, X_val, X_test

# Call the function to split the data
train_data, val_data, test_data, X_train, X_val, X_test = split_data(output_folder_path)
