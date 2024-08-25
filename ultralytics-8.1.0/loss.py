import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Function to calculate double-pass Gaussian smoothed data
def double_pass_gaussian_smooth(data, sigma1, sigma2):
    smoothed_data = gaussian_filter1d(data, sigma=sigma1, mode='reflect')
    return gaussian_filter1d(smoothed_data, sigma=sigma2, mode='reflect')

if __name__ == '__main__':
    result_dict = {
        # 'yolov3-tiny': r'../result/yolov3/results.csv',
        # 'yolov5': r'../result/yolov5/results.csv',
        # 'yolov6': r'../result/yolov5/results.csv',D:\yolo\fkz\YOLOv8-Magic-main\ultralytics-8.1.0\runs\train\yolov8-0.893
        'YOLOv8n': r'/root/autodl-tmp/fkz/YOLOv8-Magic-main/ultralytics-8.1.0/runs/train/RSOD-yolov8-0.896/results.csv',
        'YOLO-Remote': r'/root/autodl-tmp/fkz/YOLOv8-Magic-main/ultralytics-8.1.0/runs/train/RSOD-yolov8-3-0.927/results.csv',

    }

    sigma1 = 6.0  # First-pass smoothing
    sigma2 = 1.0  # Second-pass smoothing

    # Plotting map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = [float(d.strip().split()[10]) for d in datalist]
                data = np.array(data)
        x = range(len(data))
        double_smoothed_data = double_pass_gaussian_smooth(data, sigma1, sigma2)
        plt.plot(x, double_smoothed_data, label=modelname, linewidth='1')

    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid()
    plt.savefig("mAP50.png", dpi=600)
    plt.show()

    # Plotting map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[7]).values.ravel()
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = [float(d.strip().split()[11]) for d in datalist]
                data = np.array(data)
        x = range(len(data))
        double_smoothed_data = double_pass_gaussian_smooth(data, sigma1, sigma2)
        plt.plot(x, double_smoothed_data, label=modelname, linewidth='1')

    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.legend()
    plt.grid()
    plt.savefig("mAP50-95.png", dpi=600)
    plt.show()

    # Plotting total loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            box_loss = pd.read_csv(res_path, usecols=[1]).values.ravel()
            obj_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
            cls_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
            data = np.round(box_loss + obj_loss + cls_loss, 5)
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = [float(d.strip().split()[5]) for d in datalist]
                data = np.array(data)
        x = range(len(data))
        double_smoothed_data = double_pass_gaussian_smooth(data, sigma1, sigma2)
        plt.plot(x, double_smoothed_data, label=modelname, linewidth='1')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("loss.png", dpi=600)
    plt.show()
