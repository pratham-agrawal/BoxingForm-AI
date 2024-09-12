import cv2
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

for i in range(1,31):
    frame_count = count_frames_manual(f'training_videos/H_Jab/{i}.mp4')
    print(f"Manual frame count: {frame_count}")

