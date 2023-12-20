import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the pose model on the RGB frame
    results = mp_pose.process(rgb_frame)

    # Check if a person is detected
    if results.pose_landmarks is not None:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get the average of all landmark x and y coordinates to find the centroid
        x_centroid = sum([landmark.x for landmark in landmarks]) / len(landmarks)
        y_centroid = sum([landmark.y for landmark in landmarks]) / len(landmarks)

        # Draw a small dot at the centroid
        cv2.circle(frame, (int(x_centroid * frame.shape[1]), int(y_centroid * frame.shape[0])), 5, (0, 0, 255), -1)

        # Get the leftmost and rightmost coordinates of all landmarks
        x_min = min([landmark.x for landmark in landmarks])
        x_max = max([landmark.x for landmark in landmarks])

        # Get the topmost and bottommost coordinates of all landmarks
        y_min = min([landmark.y for landmark in landmarks])
        y_max = max([landmark.y for landmark in landmarks])

        # Draw a rectangle around the detected person
        cv2.rectangle(frame, (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                      (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])), (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow('Person Detection', frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("No person detected in the image")

# Release the webcam object
cap.release()
cv2.destroyAllWindows()
