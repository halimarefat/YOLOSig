import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.5),
    A.Resize(height=640, width=640, p=1.0),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Load YOLO format annotations
def load_yolo_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_labels.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:]])
    return bboxes, class_labels

# Draw bounding boxes on the image
def draw_bounding_boxes(image, bboxes, class_labels, class_names):
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    for bbox, label in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1 - 10, class_names[label], color='red', fontsize=12, weight='bold')

    plt.axis('off')
    plt.show()

# Path to your image and annotation
image_path = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/images/train/X_000_aug_0.jpeg'
annotation_path = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/labels/train/X_000_aug_0.txt'

# Load image and annotations
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes, class_labels = load_yolo_annotations(annotation_path)

# Perform augmentation
augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
augmented_image = augmented['image']
augmented_bboxes = augmented['bboxes']
augmented_class_labels = augmented['class_labels']

# Class names
class_names = ['logo', 'signature', 'routing_num', 'account_num', 'check_num_b', 'check_num_t', 'dollar_box', 'amount', 'date']

# Draw and display augmented image with bounding boxes
draw_bounding_boxes(augmented_image, augmented_bboxes, augmented_class_labels, class_names)
