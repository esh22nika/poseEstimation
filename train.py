
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os

# Paths
output_folder_path = 'preProcessedData'

# Load data
def load_data(file_paths):
    data = []
    labels = []
    for file in file_paths:
        video_data = np.load(file, allow_pickle=True)
        for seq in video_data:
            data.append(seq)
            if 'violence' in file:
                labels.append(1)  # Label for violence
            else:
                labels.append(0)  # Label for non-violence
    return np.array(data), np.array(labels)

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

    # Load datasets
    train_data, train_labels = load_data(X_train)
    val_data, val_labels = load_data(X_val)
    test_data, test_labels = load_data(X_test)

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

# Call the function to split the data
train_data, train_labels, val_data, val_labels, test_data, test_labels = split_data(output_folder_path)

# Preprocess data
max_sequence_length = max(len(seq) for seq in train_data)
train_data = pad_sequences(train_data, maxlen=max_sequence_length, padding='post', dtype='float32')
val_data = pad_sequences(val_data, maxlen=max_sequence_length, padding='post', dtype='float32')
test_data = pad_sequences(test_data, maxlen=max_sequence_length, padding='post', dtype='float32')

# Define model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(max_sequence_length, 4)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=8,
                    validation_data=(val_data, val_labels))

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save the model
model.save('pose_violence_model.h5')

