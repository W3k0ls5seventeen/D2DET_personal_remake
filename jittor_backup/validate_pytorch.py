#!/usr/bin/env python
from __future__ import division
import argparse
import os
import os.path as osp
import sys

# 导入必要的库
print("导入库...")
try:
    import numpy as np
    import cv2
    import mmcv
    import torch
    from mmcv import Config
    
    # 导入mmdet相关模块
    from mmdet import __version__
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import collect_env, get_root_logger
    from mmdet.datasets.pipelines import Compose
    from mmdet.datasets import CocoDataset
    
    print("✓ 基础库导入成功")
    
    def parse_args():
        parser = argparse.ArgumentParser(description='Validate a detector with PyTorch')
        parser.add_argument('--config', default='configs/D2Det/D2Det_detection_r50_fpn_custom.py', help='train config file path')
        parser.add_argument('--checkpoint', default='/home/w3k0lss/D2Det/work_dirs/d2det_pytorch/latest.pth', help='checkpoint file path')
        parser.add_argument('--work_dir', default='work_dirs/d2det_pytorch_val', help='the dir to save logs and results')
        parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
        args = parser.parse_args()
        return args
    
    def main():
        args = parse_args()

        # 加载配置文件
        cfg = Config.fromfile(args.config)
        cfg.work_dir = args.work_dir
        cfg.gpus = args.gpus

        # 创建工作目录
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        
        # 初始化日志
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # 记录环境信息
        meta = dict()
        env_info_dict = collect_env()
        env_info = '\n'.join([('{}: {}'.format(k, v))
                              for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # 记录基本信息
        logger.info('Config:\n{}'.format(cfg.text))
        
        # 直接创建D2Det模型实例
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        
        # 加载预训练模型
        print("\n加载预训练模型...")
        try:
            # 加载PyTorch模型
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            # 加载到模型
            model.load_state_dict(checkpoint['state_dict'])
            print("✓ 预训练模型加载成功！")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 构建验证集
        print("\n构建验证集...")
        try:
            val_dataset = build_dataset(cfg.data.val)
            print(f"✓ 验证集构建成功，包含 {len(val_dataset)} 个样本")
        except Exception as e:
            print(f"✗ 验证集构建失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 开始验证
        print("\n开始验证...")
        model.eval()
        
        # 初始化评估指标
        from mmdet.core import eval_map
        
        # 遍历验证集
        results = []
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        
        # 前向传播函数
        def forward_func(data):
            with torch.no_grad():
                result = model(**data, return_loss=False)
            return result
        
        # 遍历验证集
        for i in range(len(val_dataset)):
            if i % 100 == 0:
                print(f"验证进度: {i}/{len(val_dataset)}")
            
            # 获取数据
            data = val_dataset[i]
            # 处理数据
            processed_data = {}
            for key, value in data.items():
                if hasattr(value, 'data'):
                    value = value.data
                processed_data[key] = value
            
            # 前向传播
            result = forward_func(processed_data)
            
            results.append(result)
            
            # 收集真实标签
            gt_bboxes.append(data['gt_bboxes'].data)
            gt_labels.append(data['gt_labels'].data)
            if 'gt_bboxes_ignore' in data:
                gt_bboxes_ignore.append(data['gt_bboxes_ignore'].data)
            else:
                gt_bboxes_ignore.append(None)
        
        # 计算评估指标
        print("\n计算评估指标...")
        mean_ap, eval_results = eval_map(results, gt_bboxes, gt_labels, gt_bboxes_ignore,
                                         scale_ranges=None, iou_thr=0.5, dataset=val_dataset.CLASSES)
        
        # 打印评估结果
        print("\n评估结果:")
        print(f"mAP@0.5: {mean_ap:.4f}")
        print("\n详细结果:")
        for class_name, ap in eval_results.items():
            print(f"{class_name}: {ap:.4f}")
        
        # 保存评估结果
        import json
        result_file = osp.join(cfg.work_dir, 'eval_results.json')
        with open(result_file, 'w') as f:
            json.dump({
                'mAP': mean_ap,
                'class_results': eval_results
            }, f, indent=4)
        print(f"\n评估结果已保存到: {result_file}")
        
        # 可视化部分结果
        print("\n可视化部分结果...")
        vis_dir = osp.join(cfg.work_dir, 'visualizations')
        mmcv.mkdir_or_exist(vis_dir)
        
        # 可视化前5个结果
        for i in range(min(5, len(val_dataset))):
            # 获取图像
            img_info = val_dataset.img_infos[i]
            img_path = os.path.join(cfg.data.val.img_prefix, img_info['file_name'])
            img = cv2.imread(img_path)
            
            # 获取检测结果
            result = results[i]
            
            # 绘制检测框
            for j, bboxes in enumerate(result):
                if j >= len(val_dataset.CLASSES):
                    break
                class_name = val_dataset.CLASSES[j]
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox
                    if score > 0.5:
                        # 绘制矩形框
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # 添加标签
                        label = f"{class_name}: {score:.2f}"
                        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存可视化结果
            vis_img_path = osp.join(vis_dir, f'result_{i}.jpg')
            cv2.imwrite(vis_img_path, img)
            print(f"可视化结果保存到: {vis_img_path}")
        
        print("\n验证完成！")
        
    if __name__ == '__main__':
        main()
        
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
