import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

#Global variable
SEQUENCE_LENGTH = 15

def model_setup():
    # Set up the model
    SEQUENCE_LENGTH = 15
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 92)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Number of categories
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('form_correct2.h5')
    return model

def mp_setup():
    # Set up MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    return mp_pose, mp_drawing, pose


def classify_webcam():
    # Set up the model
    model = model_setup()

    # Set up MediaPipe Pose
    mp_pose, mp_drawing, pose = mp_setup()

    actions = ['normal', 'good jab', 'good guard']

    # Start capturing video
    cap = cv2.VideoCapture(0)

    sequence = []
    past_predictions = []
    count = 0

    while cap.isOpened():
        start_time = time.time()
        success, img = cap.read()
        count += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not success:
            break

        # Process the frame and extract pose landmarks
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            # Draw pose landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,  # Add connections between landmarks
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),  # Landmark points
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Connections
            )

            # Prepare the pose data for the model
            result_array = []
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if i in range(1, 11): continue  # Skip unwanted landmarks
                result_array.append(lm.x)
                result_array.append(lm.y)
                result_array.append(lm.z)
                result_array.append(lm.visibility)

            sequence.append(result_array)
            sequence = sequence[-SEQUENCE_LENGTH:]  # Keep the sequence to the required length

            # If sequence is long enough, make a prediction
            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # cv2.putText(img, f'Normal: {res[0]:.2f}', (70, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                # cv2.putText(img, f'Good Jab: {res[1]:.2f}', (70, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                # cv2.putText(img, f'Good Guard: {res[2]:.2f}', (70, 180), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                # Add the predicted action to past_predictions
                past_predictions.append(np.argmax(res))
                past_predictions = past_predictions[-5:]

                if np.unique(past_predictions[-5:])[0] == np.argmax(res):
                    predicted_action = actions[np.argmax(res)]
                    print(predicted_action)
                    cv2.putText(img, predicted_action, (70, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)

        # Display the frame rate and pose points
        elapsed_time = time.time() - start_time
        effective_fps = 1 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(img, f"Effective FPS: {effective_fps:.2f}", (70, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Show the video feed
        cv2.imshow("Image", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    classify_webcam()


if __name__ == "__main__":
    main()