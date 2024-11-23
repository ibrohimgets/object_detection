import cv2 as cv
import numpy as np 

# Load the Haar cascade
haar_cascade = cv.CascadeClassifier("face.xml")

# Load saved features and labels
features = np.load("features.npy")
labels = np.load("labels.npy")

# Initialize and load the trained LBPH face recognizer
face_recog = cv.face.LBPHFaceRecognizer_create()
face_recog.read("face_trained.yml")

# Define people for labels
people = ["TRUMP", "KAMALA"]

# Load and process the test image
img = cv.imread("recognation/trump/face_3.jpeg")

if img is None:
    raise FileNotFoundError("Could not read the image. Check the file path.")

# Convert to grayscale for detection
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)

# Detect faces in the image
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Loop through detected faces
for (x, y, w, h) in face_rect:
    faces_roi = gray[y:y + h, x:x + w]
    
    # Predict the label and confidence
    label, confidence = face_recog.predict(faces_roi)
    print(f"Label = {people[label]} with confidence of {confidence}")
    
    # Annotate the image
    cv.putText(img, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_COMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

print("Recognized faces!")

# Show the annotated image
cv.imshow("Recognized", img)
cv.waitKey(0)
cv.destroyAllWindows()
