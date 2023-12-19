import cv2
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

person_detected = False  # Flag to track if a person is detected

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the pose model on the RGB frame
    results = mp_pose.process(rgb_frame)

    # Check if a person is detected
    if results.pose_landmarks is not None:
        # Set the flag to True when a person is detected
        person_detected = True

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get the average of all landmark x and y coordinates to find the centroid
        x_centroid = sum([landmark.x for landmark in landmarks]) / len(landmarks)
        y_centroid = sum([landmark.y for landmark in landmarks]) / len(landmarks)

        # Print the centroid
        print(f"Person detected at centroid: ({x_centroid}, {y_centroid})")

        # Draw a small dot at the centroid
        cv2.circle(frame, (int(x_centroid * frame.shape[1]), int(y_centroid * frame.shape[0])), 5, (0, 0, 255), -1)
        cv2.imshow('Person Detection', frame)
    # Display the resulting image
    cv2.imshow('Person Detection', frame)

    # Exit the loop if 'q' key is pressed or a person is detected
    if cv2.waitKey(3) & 0xFF == ord('q') :
        break

# Release the webcam object
cap.release()
cv2.destroyAllWindows()
