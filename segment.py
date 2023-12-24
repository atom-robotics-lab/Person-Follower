'''import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

def segment_live_camera():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        if results.segmentation_mask is not None:
            frame = cv2.cvtColor(cv2.addWeighted(frame, 0.5, cv2.cvtColor(results.segmentation_mask, cv2.COLOR_GRAY2BGR), 0.5, 0), cv2.COLOR_BGR2RGB)
        cv2.imshow('Live Segmentation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
segment_live_camera()

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

def segment_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    if results.segmentation_mask is not None:
        image = cv2.cvtColor(cv2.addWeighted(image, 0.5, cv2.cvtColor(results.segmentation_mask, cv2.COLOR_GRAY2BGR), 0.5, 0), cv2.COLOR_BGR2RGB)
    return image

image_path = '/home/pragya/Downloads/gazebopose2.jpg'
segmented_image = segment_image(image_path)

cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()'''