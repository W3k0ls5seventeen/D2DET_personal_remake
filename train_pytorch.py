#!/usr/bin/env python
from __future__ import division
import argparse
import copy
import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import json

# 导入必要的库
print("导入库...")
try:
    # 导入PyTorch
    import torch
    import torchvision
    print(f"✓ PyTorch导入成功: {torch.__version__}")
    
    # 导入其他库
    import numpy as np
    import cv2
    import mmcv
    from mmcv import Config
    from mmcv.runner import init_dist, build_runner
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    
    # 导入mmdet相关模块
    from mmdet import __version__
    from mmdet.apis import set_random_seed, train_detector
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.utils import collect_env, get_root_logger
    from mmdet.datasets.pipelines import Compose
    from mmdet.datasets import CocoDataset
    
    print("✓ 基础库导入成功")
    
    # 导入模型
    sys.path.insert(0, '/home/w3k0lss/D2Det')
    # 导入PyTorch版本的D2Det
    from mmdet.models.detectors.D2Det import D2Det
    print("✓ D2Det模型导入成功")
    
    def parse_args():
        parser = argparse.ArgumentParser(description='Train a detector')
        parser.add_argument('--config', default='configs/D2Det/D2Det_detection_r50_fpn_custom.py', help='train config file path')
        parser.add_argument('--work_dir', default='work_dirs/d2det_pytorch', help='the dir to save logs and models')
        parser.add_argument(
            '--resume_from', help='the checkpoint file to resume from')
        parser.add_argument(
            '--validate',
            action='store_true',
            help='whether to evaluate the checkpoint during training')
        parser.add_argument(
            '--gpus',
            type=int,
            default=1,
            help='number of gpus to use '
            '(only applicable to non-distributed training)')
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        parser.add_argument(
            '--deterministic',
            action='store_true',
            help='whether to set deterministic options for CUDNN backend.')
        parser.add_argument(
            '--launcher',
            choices=['none', 'pytorch', 'slurm', 'mpi'],
            default='none',
            help='job launcher')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument(
            '--autoscale-lr',
            action='store_true',
            help='automatically scale lr with the number of gpus')
        args = parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)

        return args
    
    def main():
        args = parse_args()

        # 加载配置文件
        cfg = Config.fromfile(args.config)
        # 更新配置
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from
        cfg.gpus = args.gpus

        if args.autoscale_lr:
            # 应用线性缩放规则
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

        # 初始化分布式环境
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # 创建工作目录
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # 初始化日志
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
        logger.info('Distributed training: {}'.format(distributed))
        logger.info('Config:\n{}'.format(cfg.text))

        # 设置随机种子
        if args.seed is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(
                args.seed, args.deterministic))
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed
        meta['seed'] = args.seed

        # 构建模型
        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        # 构建数据集
        print('\n' + '='*60)
        print('数据集构建')
        print('='*60)
        try:
            # 首先构建原始数据集
            original_dataset = build_dataset(cfg.data.train)
            print(f'✓ 原始训练集构建成功，包含 {len(original_dataset)} 个样本')
            
            # 过滤掉不存在的图像文件
            print('\n过滤不存在的图像文件...')
            valid_indices = []
            for i in range(len(original_dataset)):
                img_info = original_dataset.img_infos[i]
                img_path = os.path.join(cfg.data.train.img_prefix, img_info['file_name'])
                if os.path.exists(img_path):
                    valid_indices.append(i)
            
            print(f'✓ 过滤完成，有效样本数: {len(valid_indices)}')
            
            # 创建一个过滤后的数据集
            from mmdet.datasets.custom import CustomDataset
            class FilteredDataset(CustomDataset):
                def __init__(self, original_dataset, valid_indices):
                    self.original_dataset = original_dataset
                    self.valid_indices = valid_indices
                    self.CLASSES = original_dataset.CLASSES
                    self.img_infos = [original_dataset.img_infos[i] for i in valid_indices]
                    self.coco = original_dataset.coco
                    self.cat_ids = original_dataset.cat_ids
                    self.cat2label = original_dataset.cat2label
                    self.img_ids = [original_dataset.img_ids[i] for i in valid_indices]
                    self.ann_file = original_dataset.ann_file
                    self.img_prefix = original_dataset.img_prefix
                    self.pipeline = original_dataset.pipeline
                    self.test_mode = original_dataset.test_mode
                    self.filter_empty_gt = original_dataset.filter_empty_gt
                    self.classwise = getattr(original_dataset, 'classwise', False)
                    # 添加flag属性，用于GroupSampler
                    if hasattr(original_dataset, 'flag'):
                        self.flag = original_dataset.flag[valid_indices]
                    else:
                        # 如果原始数据集没有flag属性，创建一个默认的
                        import numpy as np
                        self.flag = np.zeros(len(valid_indices), dtype=np.uint8)
                
                def __len__(self):
                    return len(self.valid_indices)
                
                def __getitem__(self, idx):
                    return self.original_dataset[self.valid_indices[idx]]
            
            filtered_dataset = FilteredDataset(original_dataset, valid_indices)
            datasets = [filtered_dataset]
            print(f'✓ 过滤后训练集构建成功，包含 {len(datasets[0])} 个样本')
            
            if len(cfg.workflow) == 2:
                val_dataset = copy.deepcopy(cfg.data.val)
                val_dataset.pipeline = cfg.data.train.pipeline
                datasets.append(build_dataset(val_dataset))
        except Exception as e:
            print(f'✗ 数据集构建失败: {e}')
            import traceback
            traceback.print_exc()
            return
        if cfg.checkpoint_config is not None:
            # 在检查点中保存mmdet版本、配置文件内容和类名
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=datasets[0].CLASSES)
        # 添加可视化方便的属性
        model.CLASSES = datasets[0].CLASSES
        
        # 开始训练
        logger.info('开始训练...')
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练检测器
        train_detector(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=args.validate,
            timestamp=timestamp,
            meta=meta)
        
        # 记录训练结束时间
        end_time = time.time()
        logger.info('训练完成！总耗时: {:.2f}小时'.format((end_time - start_time) / 3600))
        
        # 提取Loss曲线数据
        logger.info('提取Loss曲线数据...')
        analyze_logs(log_file, osp.join(cfg.work_dir, 'loss_curve.json'))
        logger.info('Loss曲线数据已保存到: {}'.format(osp.join(cfg.work_dir, 'loss_curve.json')))
    
    def analyze_logs(log_file, out_file):
        """从日志文件中提取Loss曲线数据"""
        import re
        
        loss_data = {
            'epoch': [],
            'loss_rpn_cls': [],
            'loss_rpn_bbox': [],
            'loss_cls': [],
            'loss_bbox': [],
            'loss_reg': [],
            'loss_mask': [],
            'total_loss': []
        }
        
        with open(log_file, 'r') as f:
            for line in f:
                # 匹配训练日志行
                if 'Epoch [' in line and 'loss:' in line:
                    # 提取epoch
                    epoch_match = re.search(r'Epoch \[([0-9]+)\]', line)
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        
                        # 提取各种loss
                        loss_rpn_cls = 0.0
                        loss_rpn_bbox = 0.0
                        loss_cls = 0.0
                        loss_bbox = 0.0
                        loss_reg = 0.0
                        loss_mask = 0.0
                        total_loss = 0.0
                        
                        # 分割行并提取各个字段
                        parts = line.split(',')
                        for part in parts:
                            part = part.strip()
                            if 'loss_rpn_cls:' in part:
                                loss_rpn_cls = float(part.split(':')[1].strip())
                            elif 'loss_rpn_bbox:' in part:
                                loss_rpn_bbox = float(part.split(':')[1].strip())
                            elif 'loss_cls:' in part:
                                loss_cls = float(part.split(':')[1].strip())
                            elif 'loss_bbox:' in part:
                                loss_bbox = float(part.split(':')[1].strip())
                            elif 'loss_reg:' in part:
                                loss_reg = float(part.split(':')[1].strip())
                            elif 'loss_mask:' in part:
                                loss_mask = float(part.split(':')[1].strip())
                            elif 'loss:' in part:
                                total_loss = float(part.split(':')[1].strip())
                        
                        # 保存数据
                        loss_data['epoch'].append(epoch)
                        loss_data['loss_rpn_cls'].append(loss_rpn_cls)
                        loss_data['loss_rpn_bbox'].append(loss_rpn_bbox)
                        loss_data['loss_cls'].append(loss_cls)
                        loss_data['loss_bbox'].append(loss_bbox)
                        loss_data['loss_reg'].append(loss_reg)
                        loss_data['loss_mask'].append(loss_mask)
                        loss_data['total_loss'].append(total_loss)
        
        # 保存到JSON文件
        with open(out_file, 'w') as f:
            json.dump(loss_data, f, indent=2)
    
    if __name__ == '__main__':
        main()
        
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
