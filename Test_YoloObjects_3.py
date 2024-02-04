import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


image_files = [
    'apple1.jpg',
    'clock_wall.jpg'
]

def object_detection_with_bounding_boxes(image_files, model="yolov3", confidence=0.2):
    for filename in image_files:
        print(f"\nDisplaying image with Image Name: {filename}")
        image_path = f"images/{filename}"
        
        # Read the image into a numpy array
        img = cv2.imread(image_path)
        
        # Perform the object detection
        bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)
        
        # Print current image's filename
        print(f"========================\nImage processed: {filename}\n")
        
        # Print detected objects with confidence level
        for l, c in zip(label, conf):
            print(f"Detected object: {l} with confidence level of {c}\n")
        
        # Create a new image that includes the bounding boxes
        output_image = draw_bbox(img, bbox, label, conf)
        
        # Save the image in the directory images_with_boxes
        cv2.imwrite(f'images_with_boxes/{filename}', output_image)
        
        # Display the image with bounding boxes
        cv2.imshow(img(filename=f'images_with_boxes/{filename}'))

# Call the function with the image_files list
object_detection_with_bounding_boxes(image_files)


