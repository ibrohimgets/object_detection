import os
import cv2 as cv
import numpy as np

# Define people and directory
people = ["trump", "kamala"]
DIR = r'C:\Users\Administrator\OneDrive\Desktop\Coding\openCV\object_detection\recognation'
haar_cascade = cv.CascadeClassifier("face.xml")

# Check if the cascade was loaded successfully
if haar_cascade.empty():
    raise FileNotFoundError("Could not load haar_cascade from face.xml")

features = []
labels = []

def create_train():
    if not os.path.exists(DIR):
        print(f"Error: The directory {DIR} does not exist.")
        return

    for new_p in people:
        path = os.path.join(DIR, new_p)
        if not os.path.exists(path):
            print(f"Warning: Directory {path} does not exist. Skipping.")
            continue

        label = people.index(new_p)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)

            if img_array is None:
                print(f"Warning: Unable to read image {img_path}. Skipping.")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                faces_roi = cv.resize(faces_roi, (100, 100))  # Resize to consistent shape
                features.append(faces_roi)
                labels.append(label)

create_train()

# Ensure there are features to train on
if len(features) == 0 or len(labels) == 0:
    raise ValueError("No faces detected. Cannot train the model.")

# Convert features to numpy array (3D array: [num_samples, height, width])
features = np.array(features, dtype="uint8")  # uint8 is the expected type for image data
labels = np.array(labels, dtype="int32")     # int32 is the expected type for labels

# Initialize LBPH face recognizer
try:
    face_recog = cv.face.LBPHFaceRecognizer_create()
except AttributeError:
    raise ImportError("cv2.face.LBPHFaceRecognizer_create is not available. Install opencv-contrib-python.")

# Train the model
face_recog.train(features, labels)
face_recog.save("face_trained.yml")

# Save features and labels
np.save("features.npy", features)
np.save("labels.npy", labels)

print("Training completed and data saved.")
