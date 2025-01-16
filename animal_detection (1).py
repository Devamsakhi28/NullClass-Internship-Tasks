


# Step 2: Import Libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
 # Use this for displaying images in Colab

# Step 3: Load YOLOv8 Model
model = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model

# Step 4: Define Carnivorous Animals and Colors
carnivorous_animals = ['cat', 'dog', 'lion', 'tiger', 'bear', 'wolf', 'leopard']
color_carnivorous = (0, 0, 255)  # Red for carnivorous animals
color_other = (0, 255, 0)  # Green for others

# Step 5: Function to Detect Animals in Images
def detect_animals_in_image(image_path):
    image = cv2.imread(image_path)  # Read the image
    results = model(image)  # Run YOLOv8 on the image
    carnivorous_count = 0  # Counter for carnivorous animals

    for result in results[0].boxes:
        class_id = int(result.cls[0])  # Class ID
        label = model.names[class_id]  # Class label
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates

        # Check if the detected animal is carnivorous
        if label in carnivorous_animals:
            color = color_carnivorous
            carnivorous_count += 1
        else:
            color = color_other

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the carnivorous count
    print(f"Number of carnivorous animals detected: {carnivorous_count}")

    # Show the processed image
    cv2.imshow('Detected Image', image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
   

# Step 6: Function to Detect Animals in Videos
def detect_animals_in_video(video_path, output_path='output.avi'):
    cap = cv2.VideoCapture(video_path)  # Open video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    carnivorous_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run YOLOv8 on the frame

        for result in results[0].boxes:
            class_id = int(result.cls[0])
            label = model.names[class_id]
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            if label in carnivorous_animals:
                color = color_carnivorous
                carnivorous_count += 1
            else:
                color = color_other

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)  # Write the processed frame to output video
        cv2_imshow(frame)  # Show frame in Colab (for demo purposes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Number of carnivorous animals detected: {carnivorous_count}")

# Step 7: Build GUI for Image/Video Input
def open_image_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    detect_animals_in_image(file_path)

def open_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    detect_animals_in_video(file_path)


# Run the GUI
def start_gui():
    root = tk.Tk()
    root.title("Animal Detection")

    btn_image = tk.Button(root, text="Detect Animals in Image", command=open_image_file)
    btn_image.pack(pady=10)

    btn_video = tk.Button(root, text="Detect Animals in Video", command=open_video_file)
    btn_video.pack(pady=10)

    root.mainloop()

# Uncomment the following line to run the GUI in a local environment
start_gui()

# For Colab, process image/video files directly using the detect functions
# Example Usage:
# Upload an image and detect
