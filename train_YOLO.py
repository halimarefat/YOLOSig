from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose a different model size: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model on the BCSD dataset
model.train(data='./bcsd.yaml', epochs=1, imgsz=640, batch=2, val=True)

# Save the trained model
model.save('./model_new.pt')
