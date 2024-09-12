import cv2
import mediapipe as mp
import numpy as np
import os
import time

def count_frames_manual(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
    cap.release()
    return frame_count

def extract_pose(video_path, save_path):
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity =1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    video_frames_count = count_frames_manual(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("could not open :", video_path)
        return
    drawing = mp.solutions.drawing_utils

    frames_data=[]
    SEQUENCE_LENGTH = 15
    #video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("TOTAL FRAMES: " + str(video_frames_count))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    duration_in_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    print(f"Duration: {duration_in_seconds}")
    skip_frames_window = max((video_frames_count/SEQUENCE_LENGTH),1)

    for frame_counter in range(SEQUENCE_LENGTH):
        print("Frame count: " + str(int(frame_counter*skip_frames_window)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_counter*skip_frames_window))
        success, img = cap.read()

        if not success or img is None:
            print(f"Frame {int(frame_counter*skip_frames_window)} could not be read.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            result_array = []
            for i, lm in enumerate(results.pose_landmarks.landmark):
                if i in range (1,11): continue
                result_array.append(lm.x)
                result_array.append(lm.y)
                result_array.append(lm.z)
                result_array.append(lm.visibility)
            frames_data.append(result_array)
            print(len(result_array))
            print(result_array)
            for i in range(0, len(result_array), 4):
                h, w, c = img.shape
                cx, cy = int(result_array[i+0] * w), int(result_array[i+1] * h)
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    while len(frames_data) < SEQUENCE_LENGTH:
        frames_data.append(frames_data[-1])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, frames_data)
    print(frames_data)

def main():
    for i in range(1,43):
        extract_pose(f'training_videos/H_Control/{i}.mp4', f'/Users/pratham/Documents/CSProjects/Form_Corrector/training_data/H_Control2/{i}')

if __name__ == "__main__":
    main()