## CSDN迪菲赫尔曼私域代码库-YOLOv8-Magic

首次使用需先配置 `github` 密钥，否则无法拉取代码!

### 配置密钥方法--[readme下面]

---

克隆代码

```
git clone https://github.com/YOLOv8-Magic/YOLOv8-Magic.git
```

```
cd YOLOv8-Magic/ultralytics-8.1.0
```

---

创建环境

```
conda create --name yolov8-magic python=3.8 -y
```

```
conda activate yolov8-magic
```

```
python -m pip install --upgrade pip
pip install -e .
```

---

推理

```
python detect.py
```

训练

```
python train.py
```

验证

```
python val.py
```

测试yaml

```
python test.py
```

导出

```
python export.py
```

---

### 配置密钥方法

```
git config --global user.name "your name"
```

```
git config --global user.email "your_email@youremail.com"
```

```
ssh-keygen -t rsa
```

```
cd ~/.ssh
```

```
cat id_rsa.pub
```

---

拉取最新代码方法

```
git config pull.rebase true # 将本地的更改移到远程分支的顶部，好像这些更改是在远程分支的更改之后进行的一样。
```

```
git pull
```
