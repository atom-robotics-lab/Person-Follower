import cv2
import numpy as np

# Function to perform basic image segmentation using thresholding
def segment_image(image_path, threshold_value=128):

    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding
    _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Apply bitwise AND to get the segmented image
    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)

    return segmented_image

# Example usage
image_path = '/home/pragya/Downloads/gazebopose2.jpg'
segmented_image = segment_image(image_path)

# Display the original and segmented images
cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
