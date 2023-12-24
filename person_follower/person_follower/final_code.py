#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import mediapipe as mp

class PersonFollower(Node):
    def __init__(self):
        super().__init__("person_follower")
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(Image, "/kinect_camera/image_raw",self.callback, 10) 
        self.depth_sub=self.create_subscription(Image,"/kinect_camera/depth/image_raw",self.depth_callback,10)
        self.velocity_msg = Twist() 
        self.depth_image = None  
    

        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def depth_callback(self,data):
       try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
       except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")


    def callback(self,data):
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic()
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        # self.cv_image= cv2.resize(self.cv_image,(720,600))
        rgb_cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.results = self.mp_pose.process(rgb_cv_image)
        if self.results.pose_landmarks is not None:
            landmarks = self.results.pose_landmarks.landmark
            # Get the average of all landmark x and y coordinates to find the centroid
            x_centroid = sum([landmark.x for landmark in landmarks]) / len(landmarks)
            y_centroid = sum([landmark.y for landmark in landmarks]) / len(landmarks)
            centroid = (x_centroid, y_centroid)
            self.x_center=x_centroid * self.cv_image.shape[1]
            self.y_center=y_centroid * self.cv_image.shape[0]
            x_length=self.cv_image.shape[1]
            self.image_center=x_length/2

            x =int(x_length/2)
            if self.depth_image is not None:
                depth_mm = self.depth_image[int(self.y_center),int(self.x_center)]
                print("depth is :",depth_mm)

            cv2.circle(self.cv_image, (int(x_centroid * self.cv_image.shape[1]), int(y_centroid * self.cv_image.shape[0])), 5, (0, 0, 255), -1)

            x_min = min([landmark.x for landmark in landmarks])
            x_max = max([landmark.x for landmark in landmarks])
            y_min = min([landmark.y for landmark in landmarks])
            y_max = max([landmark.y for landmark in landmarks])

            cv2.rectangle(self.cv_image, (int(x_min * self.cv_image.shape[1]), int(y_min * self.cv_image.shape[0])),
                          (int(x_max * self.cv_image.shape[1]), int(y_max * self.cv_image.shape[0])), (0, 255, 0), 2)
            cv2.imshow('Person Detection', self.cv_image)
            cv2.waitKey(3)
            print("Person detected in the image")
            self.results = self.holistic.process(rgb_cv_image)
            if self.results.segmentation_mask is not None:
                frame = cv2.cvtColor(cv2.addWeighted(frame, 0.5, cv2.cvtColor(results.segmentation_mask, cv2.COLOR_GRAY2BGR), 0.5, 0), cv2.COLOR_BGR2RGB)
            cv2.imshow('Live Segmentation', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            cv2.imshow('Person Detection', self.cv_image)
            cv2.waitKey(3)
            print("No person detected in the image")

def main():
  rclpy.init()
  Mynode = PersonFollower()   
  rclpy.spin(Mynode)
  cv2.destroyAllWindows()
  rclpy.shutdown()

if __name__=="__main__" :
  main()