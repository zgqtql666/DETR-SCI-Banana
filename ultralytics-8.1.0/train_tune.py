from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("ultralytics-8.1.0/yolov8n.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="ultralytics-8.1.0/ultralytics/cfg/datasets/coco128.yaml",
    epochs=30,
    iterations=300,
    optimizer="AdamW",
    plots=False,
    save=False,
    val=False,
)
