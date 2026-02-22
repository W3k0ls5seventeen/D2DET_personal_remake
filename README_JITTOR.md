# D2Det Jittor 复现版本

## 项目简介

本项目是D2Det算法的Jittor（计图）复现版本，基于mmdetection框架进行改造，专注于Jittor模型格式的训练和验证。

## 环境配置

### 1. 基础环境要求

- Python 3.7+
- Jittor 1.3.0+

### 2. 创建并激活环境

```bash
# 创建conda环境
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装Jittor

```bash
# 安装Jittor
pip install jittor

# 验证安装
python -c "import jittor as jt; print(jt.__version__)"
```

## 数据准备

### 1. COCO数据集

本项目使用COCO 2017数据集进行训练和验证。

#### 数据集结构

```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   └── train2017/
│       └── *.jpg
└── val2017/
    └── val2017/
        └── *.jpg
```

#### 下载数据集

```bash
# 下载训练集
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# 下载验证集
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# 下载标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### 2. 配置文件

修改配置文件 `configs/D2Det/D2Det_detection_r50_fpn_custom.py` 中的数据集路径：

```python
data_root = '/home/w3k0lss/D2Det/data/coco/annotations/annotations/'

img_prefix = {
    'train': '/home/w3k0lss/D2Det/data/coco/train2017/train2017/',
    'val': '/home/w3k0lss/D2Det/data/coco/val2017/val2017/'
}
```

## 模型训练

### 1. Jittor版本训练

```bash
python train_jittor.py configs/D2Det/D2Det_detection_r50_fpn_custom.py

### 2. 训练日志

训练日志会保存到 `work_dirs/d2det_jittor/` 目录下。

## 模型测试与验证

### 1. 统一验证脚本

本项目提供了统一的验证脚本，支持Jittor模型格式：

```bash
# 自动检测模型框架
python validate_unified.py

# 指定Jittor模型
python validate_unified.py --framework jittor

# 处理更多图像
python validate_unified.py --num_images 100

### 2. 官方验证脚本

```bash
# Jittor模型验证
python tools/test_jittor.py configs/D2Det/D2Det_detection_r50_fpn_custom.py \
    work_dirs/d2det_jittor/latest.pth \
    --out work_dirs/d2det_jittor/results.pkl \
    --eval bbox
```
## 项目结构

```
D2Det/
├── configs/
│   └── D2Det/
│       └── D2Det_detection_r50_fpn_custom.py
├── mmdet/
│   ├── apis/
│   ├── core/
│   ├── datasets/
│   ├── models/
│   └── ops/
├── tools/
│   └── test_jittor.py
├── train_jittor.py
├── validate_unified.py
└── requirements.txt
```
