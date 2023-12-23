#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import mediapipe as mp
import time

class PersonFollower(Node):
    def __init__(self):
        super().__init__("person_follower")
        self.bridge = CvBridge() 
        self.image_sub = self.create_subscription(Image, "/kinect_camera/image_raw",self.callback, 10) 
        self.depth_sub=self.create_subscription(Image,"/kinect_camera/depth/image_raw",self.depth_callback,10)
        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        
        self.velocity_msg = Twist() 
        self.depth_image = None  
    

        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.prev_error=0
        
    def depth_callback(self,data):
       try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
       except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")


    def callback(self,data):
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

            
            if self.depth_image is not None:
                self.depth_mm = self.depth_image[int(self.y_center),int(self.x_center)]
                print("depth is :",self.depth_mm)
                print("Person detected in the image")
                self.move_robot(self.x_center,self.y_center)
                
            else:
                self.depth_mm = 0    
            

            cv2.circle(self.cv_image, (int(x_centroid * self.cv_image.shape[1]), int(y_centroid * self.cv_image.shape[0])), 5, (0, 0, 255), -1)

            x_min = min([landmark.x for landmark in landmarks])
            x_max = max([landmark.x for landmark in landmarks])
            y_min = min([landmark.y for landmark in landmarks])
            y_max = max([landmark.y for landmark in landmarks])

            cv2.rectangle(self.cv_image, (int(x_min * self.cv_image.shape[1]), int(y_min * self.cv_image.shape[0])),
                          (int(x_max * self.cv_image.shape[1]), int(y_max * self.cv_image.shape[0])), (0, 255, 0), 2)
            self.mp_drawing.draw_landmarks(self.cv_image, self.results.pose_landmarks)
            cv2.imshow('Person Detection', self.cv_image)


            cv2.waitKey(3)
            
            print("x_centroid=", self.x_center)
            print("y_centroid=", self.y_center)
            
        else:
            cv2.imshow('Person Detection', self.cv_image)
            cv2.waitKey(3)
            print("No person detected in the image")

# prev_time= time.now();
    

    def move_robot(self, x_centroid, y_centroid):
        # Assuming your robot moves forward/backward based on x-axis and turns based on y-axis
        Kp_l = 0.09  # Kp
        Kp_a= 0.002  # Kp
        Kd_a=0.005     #Kd
        
        twist_msg = Twist()

        if self.depth_mm >= 1.4 :
            x_error = self.x_center - self.image_center  # Calculate the error from the centroid of hooman
            self.depth_mm = self.depth_image[int(self.y_center),int(self.x_center)]
               
            # Generate Twist message for robot movement
            
            P_x = Kp_l * self.depth_mm
            
            twist_msg.linear.x = P_x

            P_a = -(Kp_a * x_error)
            
            D_a=(x_error - self.prev_error)*Kd_a
            twist_msg.angular.z = P_a+D_a
            #updating the previous error
            self.prev_error=x_error
        else:
            twist_msg.linear.x = 0.0
        # Publish the Twist message
        self.velocity_publisher.publish(twist_msg)


def main():
  rclpy.init()
  Mynode = PersonFollower()   
  rclpy.spin(Mynode)
  cv2.destroyAllWindows()
  rclpy.shutdown()

if __name__=="__main__" :
  main()