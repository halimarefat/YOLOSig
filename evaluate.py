from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO('./model.pt')
model.model.names = {0: 'logo', 1: 'signature', 2: 'routing_num', 3: 'account_num', 4: 'check_num_b', 5: 'check_num_t', 6: 'dollar_box', 7: 'amount', 8: 'date'}

print(model.names)

# Load the image
image_path = '/Users/ali/Desktop/BCSD/TestSet/X/X_016.jpeg'  # Update with your image path
image = cv2.imread(image_path)

# Run inference
results = model.predict(source=image_path, conf=0.07)
print(len(results))
# Print results
#print(f'The result is here: {results}')

# Check if there are any detections
if len(results) > 0:
    result = results[0]  # Get the first result (there should only be one for a single image)

    # Draw bounding boxes if there are any detections
    if result.boxes is not None:
        boxes = result.boxes.xyxy  # Get bounding box coordinates
        confidences = result.boxes.conf  # Get confidence scores
        class_ids = result.boxes.cls  # Get class IDs

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(class_id)]}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else:
        print("No detections")

# Convert BGR image to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
