import tensorflow as tf
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model('pose_violence_model.h5')

# Define the paths to the test data
output_folder_path = 'preProcessedData'
X_test_file_paths = [os.path.join(output_folder_path, file) for file in os.listdir(output_folder_path) if 'test' in file]
y_test = [1 if 'violence' in file else 0 for file in os.listdir(output_folder_path) if 'test' in file]  # Label for test data

# Load test data
def load_data(file_paths):
    data = [np.load(file) for file in file_paths]
    return np.array(data)

X_test = load_data(X_test_file_paths)

# Ensure the input shape matches the model's expected input shape
X_test = [np.array(np.pad(x, ((0, max(0, 33 - x.shape[0])), (0, 0)), mode='constant')) for x in X_test]
X_test = np.array(X_test)

# Ensure y_test is a numpy array
y_test = np.array(y_test)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
