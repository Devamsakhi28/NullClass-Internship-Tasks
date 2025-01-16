import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Load the models
emotion_model = tf.keras.models.load_model('FER_model.h5')
age_model = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')

# Age and Emotion categories
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(13-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Age Detection Function
def detect_age(face_image, model):
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
    model.setInput(blob)
    predictions = model.forward()
    age_index = predictions[0].argmax()
    return AGE_LIST[age_index]

# Emotion Detection Function
def detect_emotion(face_image, model):
    # Convert the face image to grayscale
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Resize to the model's input size (48x48)
    resized_face = cv2.resize(gray_face, (48, 48))

    # Normalize and reshape
    normalized_face = resized_face / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    input_face = np.expand_dims(input_face, axis=-1)

    # Predict the emotion
    predictions = model.predict(input_face)
    emotion_index = np.argmax(predictions)
    return EMOTION_LIST[emotion_index]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for webcam
data = []  # To store age, emotion, and entry time

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]

        # Detect age
        try:
            age = detect_age(face_image, age_model)
        except Exception as e:
            print(f"Error in age detection: {e}")
            continue

        # Check age and determine action
        if '(0-2)' in age or '(8-12)' in age or '(60-100)' in age:
            # Mark as not allowed
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
            cv2.putText(frame, "Not Allowed", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            data.append([age, 'N/A', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        else:
            # Detect emotion
            try:
                emotion = detect_emotion(face_image, emotion_model)
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                continue

            # Mark as allowed
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, f"{age}, {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            data.append([age, emotion, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    # Display the video
    cv2.imshow("Age and Emotion Detection", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save data to CSV
if data:
    df = pd.DataFrame(data, columns=['Age', 'Emotion', 'Entry Time'])
    df.to_csv('movie_theater_data.csv', index=False)
    print("Data saved to movie_theater_data.csv")
else:
    print("No data to save.")

