#! /usr/bin/env python3

import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,'/kinect_camera/image_raw', self.image_callback, 10)
        self.cv_image=None
        self.bridge = CvBridge()
        base_options = python.BaseOptions(model_asset_path='deeplabv3.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.image_subscription = self.create_subscription(Image, "/kinect_camera/image_raw", self.image_callback, 10)
    
    #take webcam feed and display 
    def calculate_centroid(self,keypoints):
        x_sum, y_sum = 0, 0
        num_points = len(keypoints)
        for point in keypoints:
            x_sum += point[0]
            y_sum += point[1]
        centroid_x = x_sum / num_points
        centroid_y = y_sum / num_points
        return [int(centroid_x), int(centroid_y)]

    def webcam_centroid(self):
        #while True:
        #self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        try:
            mp_holistic = mp.solutions.holistic
            mp_drawing = mp.solutions.drawing_utils
            holistic = mp_holistic.Holistic()
            results = holistic.process(self.cv_image)
            if results.pose_landmarks:
                #keypoints for centroid
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    h, w, c = self.cv_image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    keypoints.append((x, y))
                centroid = self.calculate_centroid(keypoints) #centroid calculation
                cv2.circle(self.cv_image, centroid, 5, (0, 255, 0), -1)
                mp_drawing.draw_landmarks(self.cv_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            cv2.imshow('Kinect Camera Feed', self.cv_image)
            #cv2.waitkey(3)
        except Exception as e:
            print(e)
        #if cv2.waitKey(3) & 0xFF == ord('q'):
            #break
        #self.cv_image.release()
        #cv2.destroyAllWindows()
    
    def segmentation(self):
        try:
            #cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            image = mp.Image(frame_rgb)
            # Retrieve the masks for the segmented image
            segmentation_result = self.segmenter.segment(image)
            category_mask = segmentation_result.category_mask
            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
            # Display the original and segmented images
            cv2.imshow('Original Image', self.cv_image)
            cv2.imshow('Segmented Image', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except Exception as e:
            print(e)

    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        #cv2.imshow('Kinect Camera Feed', self.cv_image)
        self.webcam_centroid()
        cv2.waitKey(3)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    #camera_subscriber.webcam_centroid()
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
