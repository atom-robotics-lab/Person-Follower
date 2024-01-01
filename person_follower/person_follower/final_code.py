#! /usr/bin/env python3

import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/kinect_camera/image_raw', self.image_callback, 10)
        self.cv_image = None
        self.mp_holistic = mp.solutions.holistic.Holistic()
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_centroid(self, keypoints):
        x_sum, y_sum = 0, 0
        num_points = len(keypoints)
        for point in keypoints:
            x_sum += point[0]
            y_sum += point[1]
        centroid_x = x_sum / num_points
        centroid_y = y_sum / num_points
        return [int(centroid_x), int(centroid_y)]

    def perform_segmentation(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, segmented_mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        return segmented_mask

    def webcam_centroid(self):
        try:
            results = self.mp_holistic.process(self.cv_image)
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    h, w, c = self.cv_image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    keypoints.append((x, y))
                centroid = self.calculate_centroid(keypoints)
                cv2.circle(self.cv_image, centroid, 5, (0, 255, 0), -1)
                self.mp_drawing.draw_landmarks(self.cv_image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

            #image segmentation
            segmented_mask = self.perform_segmentation(self.cv_image)
            cv2.imshow('Segmentation Mask', segmented_mask)
            cv2.imshow('Kinect Camera Feed', self.cv_image)

        except Exception as e:
            print(e)

    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.webcam_centroid()
        cv2.waitKey(3)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.mp_holistic.close()  
    cv2.destroyAllWindows()  
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

