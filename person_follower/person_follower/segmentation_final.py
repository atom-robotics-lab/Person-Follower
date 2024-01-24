#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class PersonFollower(Node):
    def __init__(self):
        super().__init__("person_follower")
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(Image, "/kinect_camera/image_raw",self.callback, 10) 
        self.depth_sub=self.create_subscription(Image,"/kinect_camera/depth/image_raw",self.depth_callback,10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.velocity_msg = Twist()
        self.depth_image = None
        self.velocity_msg.linear.y = 0.0
        self.velocity_msg.linear.z = 0.0
        self.velocity_msg.angular.x = 0.0
        self.velocity_msg.angular.y = 0.0

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.kp_angular=0.005
        self.kp_linear=0.7
        self.x_center=None
        self.image_center=None
        self.buffer=10
        
        # Create the options that will be used for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path='/home/pragya/personfollower_ws/src/Person-Follower/person_follower/person_follower/deeplabv3.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options,output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

        self.BG_COLOR = (192, 192, 192) # gray
        self.MASK_COLOR = (0, 255, 0) # white

    def depth_callback(self,data):
       try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
       except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")


    def callback(self,data):
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        # self.cv_image= cv2.resize(self.cv_image,(720,600))
        rgb_cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.segmentation_frame=self.cv_image
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
            # if self.depth_image is not None:
            #     depth_mm = self.depth_image[int(self.y_center),int(self.x_center)]
            #     print("depth is :",depth_mm)

            cv2.circle(self.cv_image, (int(x_centroid * self.cv_image.shape[1]), int(y_centroid * self.cv_image.shape[0])), 5, (0, 0, 255), -1)

            x_min = min([landmark.x for landmark in landmarks])
            x_max = max([landmark.x for landmark in landmarks])
            y_min = min([landmark.y for landmark in landmarks])
            y_max = max([landmark.y for landmark in landmarks])

            cv2.rectangle(self.cv_image, (int(x_min * self.cv_image.shape[1]), int(y_min * self.cv_image.shape[0])),
                          (int(x_max * self.cv_image.shape[1]), int(y_max * self.cv_image.shape[0])), (0, 255, 0), 2)
            self.segmentation_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.segmentation_frame)
            # mask for the segmented image
            segmentation_result = self.segmenter.segment(self.segmentation_frame)
            category_mask = segmentation_result.category_mask

            image_data = self.segmentation_frame.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = self.MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = self.BG_COLOR
            
            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2

            self.segmentation_frame = np.where(condition, fg_image, bg_image)
            self.mp_drawing.draw_landmarks(self.segmentation_frame, self.results.pose_landmarks,self.mp_holistic.POSE_CONNECTIONS)

            if self.results.pose_landmarks is not None:
                cv2.line(self.cv_image, (int(self.x_center-15), int(self.y_center)), (int(self.x_center+15), int(self.y_center)), (255, 0, 0), 3) 
                cv2.line(self.cv_image, (int(self.x_center), int(self.y_center-15)), (int(self.x_center), int(self.y_center+15)), (255, 0, 0), 3)
                cv2.line(self.segmentation_frame, (int(self.x_center-15), int(self.y_center)), (int(self.x_center+15), int(self.y_center)), (255, 0, 0), 3) 
                cv2.line(self.segmentation_frame, (int(self.x_center), int(self.y_center-15)), (int(self.x_center), int(self.y_center+15)), (255, 0, 0), 3)
                cv2.line(self.segmentation_frame, (int(350), int(0)), (int(350), int(500)), (0, 0, 255), 2) 
                cv2.line(self.segmentation_frame, (int(0), int(self.y_center)), (int(700), int(self.y_center)), (0, 0, 255), 2)
                

        self.control_loop()
    
    def control_loop(self):        
        if self.results.pose_landmarks is  None:
            self.velocity_control(0.0,-0.5)
            self.top="Searching for Person"
            self.bottom="<=Rotating=>"
        
        else:
            if self.x_center > 700:
                self.x_center=699.0
            if self.y_center> 500:
                self.y_center=499.0
            depth_mm = self.depth_image[int(self.y_center),int(self.x_center)]

            if (3.0 + self.x_center) < self.image_center :  #person is left means positive ang vel
                self.top="<==Left"
                self.bottom="Going Forward"
 
                self.error_linear= depth_mm - 3.0
                
                self.error= (self.image_center - self.x_center)
                self.angular_vel= self.kp_angular * self.error
                if self.error_linear < 0.3:
                    self.error_linear=0.0
                self.linear_vel= self.kp_linear * self.error_linear
                self.velocity_control(self.linear_vel,self.angular_vel)

            elif self.x_center > self.image_center + 3.0 :  #person is right means negative ang vel
                self.top="Right==>"
                self.bottom="Going Right"

                self.error_linear= depth_mm - 3.0
                self.error= (self.image_center - self.x_center)
                self.angular_vel= self.kp_angular * self.error

                if self.error_linear < 0.3:
                    self.error_linear=0.0
                self.linear_vel= self.kp_linear * self.error_linear
                self.velocity_control(self.linear_vel,self.angular_vel)
            
            elif abs(self.x_center - self.image_center)<= 3.0 :  
                self.top="Stop"
                self.bottom="Reached Goal position"
                self.velocity_control(0.0,0.0)


        cv2.putText(self.cv_image,self.top,(200,50),cv2.FONT_HERSHEY_DUPLEX,0.8,(0, 0,255),2)    
        cv2.putText(self.cv_image,self.bottom,(200,450),cv2.FONT_HERSHEY_DUPLEX,0.8,(0, 0, 255),2)                      
        cv2.imshow('Person Detection', self.cv_image)
        cv2.imshow('Person Segmentation', self.segmentation_frame)
        cv2.waitKey(3)
        print("No person detected in the image")

    def velocity_control(self,linear,angular):
        # linear = min(linear,2) 
        self.velocity_msg.linear.x = float(linear)
        self.velocity_msg.angular.z = angular
        self.pub.publish(self.velocity_msg)  

def main():
  rclpy.init()
  Mynode = PersonFollower()   
  rclpy.spin(Mynode)
  cv2.destroyAllWindows()
  rclpy.shutdown()

if __name__=="__main__" :
  main()