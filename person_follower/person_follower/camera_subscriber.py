#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber_node')
        self.subscription = self.create_subscription(
            Image,'/kinect_camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        print("holaaaaaaaa")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, encoding='bgr8')
        cv2.imshow('Kinect Camera Feed', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()
    rclpy.spin(camera_subscriber)
    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()