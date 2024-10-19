import cv2
import numpy as np
import os

def create_yolov8_seg_label(mask_path, image_path, output_label_path):
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask from {mask_path}")
        return
    
    # Load the corresponding image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return
    img_height, img_width = image.shape[:2]
    
    # Dilate the mask to ensure the bounding box covers the entire signature
    kernel = np.ones((10, 10), np.uint8)  # Increased kernel size
    dilated_mask = cv2.dilate(mask, kernel, iterations=3)
    
    # Find contours in the dilated mask image
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No contours found in {mask_path}.")
        return
    
    # Assuming the largest contour is the signature
    contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    
    # Get segmentation points
    segmentation_points = contour.flatten().tolist()
    normalized_points = [point / img_width if i % 2 == 0 else point / img_height for i, point in enumerate(segmentation_points)]
    
    # Construct label line
    class_id = 0  # Replace with your actual class ID if different
    label_line = [class_id, center_x, center_y, norm_w, norm_h] + normalized_points
    
    # Save YOLOv8 segmentation label
    with open(output_label_path, 'w') as label_file:
        label_file.write(" ".join(map(str, label_line)))
    
    print(f"Saved YOLOv8 segmentation label to {output_label_path}")

def draw_yolo_segmentation(image_path, label_path, output_image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return
    
    # Read the YOLO label file
    if not os.path.exists(label_path):
        print(f"Label file {label_path} does not exist")
        return
    
    with open(label_path, 'r') as file:
        label = file.readline().strip().split()
    
    if len(label) < 6:
        print(f"Label file {label_path} is not in YOLO segmentation format")
        return
    
    # Extract the bounding box and segmentation points from the label
    class_id = int(label[0])
    center_x = float(label[1])
    center_y = float(label[2])
    norm_w = float(label[3])
    norm_h = float(label[4])
    segmentation_points = list(map(float, label[5:]))
    
    img_height, img_width = image.shape[:2]
    
    # Convert normalized coordinates to pixel values
    x = int((center_x - norm_w / 2) * img_width)
    y = int((center_y - norm_h / 2) * img_height)
    w = int(norm_w * img_width)
    h = int(norm_h * img_height)
    
    # Draw the bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Convert segmentation points to pixel values and draw the polygon
    points = np.array(segmentation_points).reshape(-1, 2)
    points[:, 0] *= img_width
    points[:, 1] *= img_height
    points = points.astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Save the image with the bounding box and polygon
    cv2.imwrite(output_image_path, image)
    print(f"Saved image with bounding box and segmentation polygon to {output_image_path}")

mask_path = '../BCSD/TrainSet/y/y_002.jpeg'  # Update this path to the correct location of your mask
image_path = '../BCSD/TrainSet/images/X_002.jpeg'  # Update this path to the corresponding image
output_label_path = './X_002.txt'  # Update this path to save the label
output_image_path = './X_002_with_bb.jpeg'  # Update this path to save the drawn image

os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Create YOLOv8 segmentation label
create_yolov8_seg_label(mask_path, image_path, output_label_path)

# Draw the bounding box and segmentation polygon on the image
draw_yolo_segmentation(image_path, output_label_path, output_image_path)
