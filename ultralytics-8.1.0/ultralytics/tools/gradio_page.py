# pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
import gradio as gr
from ultralytics import YOLO
import PIL.Image as Image

model = YOLO("yolov8n.pt")


def predict_image(img, conf_threshold, iou_threshold):

    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
    examples=[
        ["ultralytics/ultralytics/assets/bus.jpg", 0.25, 0.45],
        ["ultralytics/ultralytics/assets/zidane.jpg", 0.25, 0.45],
    ],
)

if __name__ == "__main__":
    iface.launch()
