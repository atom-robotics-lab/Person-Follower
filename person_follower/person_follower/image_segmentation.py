import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

class ImageSegmentationNode(Node):
    def __init__(self):
        self.bridge = CvBridge()

        # Create the options that will be used for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path='deeplabv3.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

        # Create the image segmenter
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

        # Subscribe to the image topic
        self.image_subscription = self.create_subscription(Image, "/kinect_camera/image_raw", self.image_callback, 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
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
            cv2.imshow('Original Image', cv_image)
            cv2.imshow('Segmented Image', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


