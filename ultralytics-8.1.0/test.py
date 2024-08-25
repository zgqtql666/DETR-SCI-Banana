
from ultralytics import YOLO
# 加载训练好的模型或者网络结构配置文件
model = YOLO('/root/autodl-tmp/fkz/YOLOv8-Magic-main/ultralytics-8.1.0/runs/train/yolov8-GRF_SPPF-SaELayer-FocalerMDPIOU-0.919/weights/best.pt')
print(model.info(detailed=True))

# model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
