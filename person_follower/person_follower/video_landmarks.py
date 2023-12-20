import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5

)
mp_drawing = mp.solutions.drawing_utils


# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run MediaPipe Pose model on the RGB frame
    results = mp_pose.process(rgb_frame)

    # Draw landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks)

    # Display the resulting image
    cv2.imshow('Person Detection', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
