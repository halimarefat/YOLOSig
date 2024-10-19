import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_file, classes, output_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        class_id = classes.index(class_name)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

def convert_annotations(input_dir, output_dir, classes):
    os.makedirs(output_dir, exist_ok=True)
    for xml_file in os.listdir(input_dir):
        if xml_file.endswith('.xml'):
            convert_xml_to_yolo(os.path.join(input_dir, xml_file), classes, output_dir)

# Define your classes here
classes = ['logo', 'signature', 'routing_num', 'account_num', 'check_num_b', 'check_num_t', 'dollar_box', 'amount', 'date']

# Input directory containing XML files
input_dir = '/Users/ali/Desktop/YOLOSeg/labels/train'

# Output directory to save YOLO format annotations
output_dir = '/Users/ali/Desktop/YOLOSeg/BCSD_YOLO/labels/train'

convert_annotations(input_dir, output_dir, classes)
