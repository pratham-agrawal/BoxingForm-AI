import cv2
import mediapipe as mp

pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    model_complexity =1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing = mp.solutions.drawing_utils
img = cv2.imread('full.jpg', 1)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = pose.process(imgRGB)


print(result.pose_landmarks)

if result.pose_landmarks:
    drawing.draw_landmarks(
        image = img,
        landmark_list= result.pose_landmarks,
        connections = mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec= drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=7,
            circle_radius=4
        ),
        connection_drawing_spec= drawing.DrawingSpec(
            color=(0, 0, 255),
            thickness = 11,
            circle_radius=3
        )
    )
cv2.imshow("Pose Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
