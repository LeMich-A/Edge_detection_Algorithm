import cv2
import numpy as np
import os
"""import cvlib as cv
from cvlib.object_detection import draw_bbox"""

image_files = [
    'apple1.jpg',
    'clock_wall.jpg'
]

for image_file in image_files:
    print(f"\nDisplaying image with Image Name: {image_file}")
    
    # Load the image using cv2.imread
    image_path = f"images/{image_file}"
    img = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if img is not None:
        # Display the image 
        cv2.imshow(f"Image: {image_file}", img)
        
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error loading image: {image_file}")
   

 
    dir_name = "images_with_boxes"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)