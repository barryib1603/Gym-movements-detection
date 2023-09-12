import tkinter as tk
import customtkinter as ck
import pandas as pd
import numpy as np
import pickle
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks

# Setup the main window
mainWindow = tk.Tk()
mainWindow.geometry("480x700")
mainWindow.title("Swoleboi")

# Apply dark mode
ck.set_appearance_mode("dark")

# Setup labels for stage, reps, and probability
stageLabel = ck.CTkLabel(mainWindow, height=40, width=120, font=("Arial", 20), text_color="white", padx=10)
stageLabel.place(x=10, y=1)
stageLabel.configure(text='STAGE')

repsLabel = ck.CTkLabel(mainWindow, height=40, width=120, font=("Arial", 20), text_color="white", padx=10)
repsLabel.place(x=300, y=1)
repsLabel.configure(text='REPS')

probabilityLabel = ck.CTkLabel(mainWindow, height=40, width=120, font=("Arial", 20), text_color="white", padx=10)
probabilityLabel.place(x=160, y=1)
probabilityLabel.configure(text='PROB')

# Box to display values
stageValueBox = ck.CTkLabel(mainWindow, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
stageValueBox.place(x=10, y=41)
stageValueBox.configure(text='0')

repsValueBox = ck.CTkLabel(mainWindow, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
repsValueBox.place(x=300, y=41)
repsValueBox.configure(text='0')

probabilityValueBox = ck.CTkLabel(mainWindow, height=40, width=120, font=("Arial", 20), text_color="white",
                                  fg_color="blue")
probabilityValueBox.place(x=160, y=41)
probabilityValueBox.configure(text='0')


# Function to reset counter
def reset_rep_count():
    global repCounter
    repCounter = 0


# Button to reset the counter
resetButton = ck.CTkButton(mainWindow, text='RESET', command=reset_rep_count, height=40, width=120, font=("Arial", 20),
                           text_color="white", fg_color="blue")
resetButton.place(x=10, y=600)

# Video frame for the feed
videoFrame = tk.Frame(mainWindow, height=480, width=480, bg="black")
videoFrame.place(x=10, y=90)
videoLabel = tk.Label(videoFrame, bg="black")
videoLabel.place(x=0, y=0)

# Initialize MediaPipe drawing and pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
poseEstimation = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Load the model for prediction
with open('deadlift.pkl', 'rb') as file:
    trainedModel = pickle.load(file)

videoCapture = cv2.VideoCapture(0)
currentPose = ''
repCounter = 0
bodyLanguageProbability = np.array([0, 0])
bodyLanguageClass = ''


# Function to detect pose and count reps
def detect_pose():
    global currentPose
    global repCounter
    global bodyLanguageClass
    global bodyLanguageProbability

    _, capturedFrame = videoCapture.read()
    rgbImage = cv2.cvtColor(capturedFrame, cv2.COLOR_BGR2RGB)
    poseResults = poseEstimation.process(rgbImage)

    # Draw landmarks on the image
    mp_drawing.draw_landmarks(rgbImage, poseResults.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                              mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

    try:
        dataRow = np.array([[result.x, result.y, result.z, result.visibility] for result in
                            poseResults.pose_landmarks.landmark]).flatten().tolist()
        dataFrame = pd.DataFrame([dataRow], columns=landmarks)
        bodyLanguageProbability = trainedModel.predict_proba(dataFrame)[0]
        bodyLanguageClass = trainedModel.predict(dataFrame)[0]

        if bodyLanguageClass == "down" and bodyLanguageProbability[bodyLanguageProbability.argmax()] > 0.7:
            currentPose = "down"
        elif currentPose == "down" and bodyLanguageClass == "up" and bodyLanguageProbability[
            bodyLanguageProbability.argmax()] > 0.7:
            currentPose = "up"
            repCounter += 1

    except Exception as error:
        print(error)

    displayImage = rgbImage[:, :460, :]
    imageArray = Image.fromarray(displayImage)
    tkImage = ImageTk.PhotoImage(imageArray)
    videoLabel.imgtk = tkImage
    videoLabel.configure(image=tkImage)
    videoLabel.after(10, detect_pose)

    repsValueBox.configure(text=repCounter)
    probabilityValueBox.configure(text=bodyLanguageProbability[bodyLanguageProbability.argmax()])
    stageValueBox.configure(text=currentPose)


detect_pose()
mainWindow.mainloop()
