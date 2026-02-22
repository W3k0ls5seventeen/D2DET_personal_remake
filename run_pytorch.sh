#!/bin/bash

# 激活open-mmlab环境
echo "激活open-mmlab环境..."
source activate open-mmlab

# 安装必要的依赖
echo "安装必要的依赖..."
pip install -r requirements/runtime.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行PyTorch版本的训练脚本
echo "运行PyTorch版本的训练脚本..."
python train_pytorch.py
