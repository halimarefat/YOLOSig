import os
import cv2
import albumentations as A

def augment_and_save(image, bboxes, class_labels, output_image_dir, output_label_dir, image_name, num_augmentations=5):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RGBShift(p=0.5),
        A.Resize(height=560, width=1120, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for i in range(num_augmentations):
        print(f'num is {i}')
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']

            output_image_path = os.path.join(output_image_dir, f"{image_name}_aug_{i}.jpeg")
            output_label_path = os.path.join(output_label_dir, f"{image_name}_aug_{i}.txt")

            cv2.imwrite(output_image_path, augmented_image)
            with open(output_label_path, 'w') as f:
                for bbox, label in zip(augmented_bboxes, class_labels):
                    f.write(f"{label} {' '.join(map(str, bbox))}\n")
            print(f"Saved augmented image: {output_image_path}")
            print(f"Saved augmented label: {output_label_path}")
        except Exception as e:
            print(f"Error during augmentation and save: {e}")

def load_yolo_annotations(annotation_path):
    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        class_labels = []
        for line in lines:
            parts = line.strip().split()
            class_labels.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])
        return bboxes, class_labels
    except Exception as e:
        print(f"Error loading annotations from {annotation_path}: {e}")
        return [], []

def augment_dataset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, num_augmentations=5):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    for image_name in os.listdir(input_image_dir):
        if image_name.endswith('.jpeg'):
            try:
                image_path = os.path.join(input_image_dir, image_name)
                label_path = os.path.join(input_label_dir, os.path.splitext(image_name)[0] + '.txt')
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image {image_path}")
                    continue
                bboxes, class_labels = load_yolo_annotations(label_path)
                augment_and_save(image, bboxes, class_labels, output_image_dir, output_label_dir, os.path.splitext(image_name)[0], num_augmentations)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

if __name__ == "__main__":
    # Define your input and output directories
    input_image_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/images/train'
    input_label_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/labels/train'
    output_image_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/images/augmented'
    output_label_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/labels/augmented'

    # Number of augmentations per image
    num_augmentations = 10

    # Augment the dataset
    print('augmentation is started!')
    augment_dataset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, num_augmentations)


