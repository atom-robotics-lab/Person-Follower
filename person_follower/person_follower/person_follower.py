#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import mediapipe as mp

class PersonFollower(Node):
    def __init__(self):
        super().__init__("person_follower")
        self.bridge = CvBridge()

        # Subscribe to the RGB image topic and depth image topic
        self.image_sub = self.create_subscription(Image, "/kinect_camera/image_raw", self.callback, 10)
        self.depth_sub = self.create_subscription(Image, "/kinect_camera/depth/image_raw", self.depth_callback, 10)

        # Publisher for robot velocity commands
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Initialize variables
        self.last_error = 0.0
        self.last_depth = 0.0
        self.depth_image = None
        self.person_detected = False

        # Initialize MediaPipe Pose module
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def depth_callback(self, data):
        try:
            # Convert depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def callback(self, data):

        # Convert RGB image to OpenCV format
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        rgb_cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)

        # Process the pose using MediaPipe
        self.results = self.mp_pose.process(rgb_cv_image)

        if self.results.pose_landmarks is not None:
            # Person detected
            self.person_detected = True
            landmarks = self.results.pose_landmarks.landmark

            # Calculate centroid
            x_centroid = sum([landmark.x for landmark in landmarks]) / len(landmarks)
            y_centroid = sum([landmark.y for landmark in landmarks]) / len(landmarks)
            self.x_center = x_centroid * self.cv_image.shape[1]
            self.y_center = y_centroid * self.cv_image.shape[0]
            self.image_center = self.cv_image.shape[1] / 2

            # Setting the limit for co-ordinates
            self.limiting_loop()

            # Check if depth information is available
            if self.depth_image is not None:
                self.depth_mm = self.depth_image[int(self.y_center), int(self.x_center)]
                print("Depth is:", self.depth_mm)
            else:
                self.vel_control(0.0, 0.0)

            # Draw landmarks and bounding box
            self.draw_landmarks_and_box()

            # Move the robot based on the detected person
            self.move_robot()

            # Display the image
            cv2.imshow('Person Detection', self.cv_image)
            cv2.waitKey(3)

        else: # No Person Detected

            if self.person_detected == True:  # Person was detected atleast once before
                self.vel_control(0.0 , -0.5)

                # Put text on the screen
                top = "Searching Person"
                bottom = "Stop"
                self.display_text_on_image(bottom,top)
            
            else:  # Person was never detected
                self.vel_control(0.0, 0.0)

    def draw_landmarks_and_box(self):
        cv2.circle(self.cv_image, (int(self.x_center), int(self.y_center)), 5, (0, 0, 255), -1)

        x_min = min([landmark.x for landmark in self.results.pose_landmarks.landmark])
        x_max = max([landmark.x for landmark in self.results.pose_landmarks.landmark])
        y_min = min([landmark.y for landmark in self.results.pose_landmarks.landmark])
        y_max = max([landmark.y for landmark in self.results.pose_landmarks.landmark])

        cv2.rectangle(self.cv_image, (int(x_min * self.cv_image.shape[1]), int(y_min * self.cv_image.shape[0])),
                      (int(x_max * self.cv_image.shape[1]), int(y_max * self.cv_image.shape[0])), (0, 255, 0), 2)
        self.mp_drawing.draw_landmarks(self.cv_image, self.results.pose_landmarks)

    def move_robot(self):
        # Constants for PD controller
        Kp_l = 0.58
        Kp_yaw = 0.005
        Kd_yaw = 0.00265
        Kd_l = 0.25

        # Calculating the error
        x_error = self.x_center - self.image_center

        if self.depth_mm > 1.5:

            # Determine the direction to move based on the person's position
            if x_error > 10:
                top = "Right==>"
                bottom = "Go Forward"
            elif x_error < -10:
               top = "<==Left"
               bottom = "Go Forward"
            else:
               top = "Centre"
               bottom = "Go Forward"
        else:
            # Stop the robot if depth information is insufficient
            self.vel_control(0.0, 0.0)
            top = "Centre"
            bottom = "Go Forward"

        # Proportional and Derivative Drive
        P_x = Kp_l * self.depth_mm
        P_yaw = -(Kp_yaw * x_error)
        D_yaw = ((x_error - self.last_error) / 0.6) * Kd_yaw
        D_l = ((self.depth_mm - self.last_depth) / 0.6) * Kd_l

        self.last_depth = self.depth_mm
        self.last_error = x_error

        # Publish the Twist message to move the robot
        self.vel_control((P_x + D_l), (P_yaw + D_yaw))

        # Display text on the image
        self.display_text_on_image(bottom, top)
        

    
    def limiting_loop(self):
        if self.x_center > 700:
                self.x_center=699.0
        if self.y_center > 500:
                self.y_center=499.0
    
    def vel_control(self, vel_x, vel_spin):
        # Search for person in the vicinity
        twist_msg = Twist()
        twist_msg.linear.x = vel_x
        twist_msg.angular.z = vel_spin
        self.velocity_publisher.publish(twist_msg)

    def display_text_on_image(self, bottom, top):
        # Display text on the image
        img = self.cv_image
        text1 =bottom
        text2 =top
        txt1_location = (300, 450)
        txt2_location = (300, 30)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        fontScale = 1
        fontColor_lt = (255, 255, 255)
        fontColor_at = (0, 100, 0)
        thickness = 1
        lineType = cv2.LINE_AA

        cv2.putText(img, text1, txt1_location, font, fontScale, fontColor_lt, thickness, lineType)
        cv2.putText(img, text2, txt2_location, font, fontScale, fontColor_at, thickness, lineType)

def main():
    rclpy.init()
    Mynode = PersonFollower()
    rclpy.spin(Mynode)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
