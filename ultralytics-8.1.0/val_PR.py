# 参考博客：https://blog.csdn.net/ljlqwer/article/details/129175087
from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/fkz/YOLOv8-Magic-main/ultralytics-8.1.0/runs/train/RT-DETR-0.828/weights/best.pt")  # 权重地址

results = model.val(data="/root/autodl-tmp/fkz/YOLOv8-Magic-main/datasets/NWPU VHR-10 dataset/NWPU.yaml", imgsz=640, split='val', batch=1, conf=0.001, iou=0.6, name='yolov8s-from-ultralytics-main-bs1', optimizer='Adam')  # 参数和训练用到的一样
