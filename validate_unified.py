#!/usr/bin/env python
from __future__ import division
import argparse
import os
import os.path as osp
import sys
import json
import time

print("导入基础库...")
try:
    import numpy as np
    import cv2
    import torch
    from mmcv import Config
    import mmcv

    print("✓ 基础库导入成功")

    print("\n检查PyTorch...")
    pytorch_available = True
    print(f"✓ PyTorch版本: {torch.__version__}")

    print("\n尝试导入mmdet核心模块...")
    mmdet_available = False
    try:
        from mmdet.apis import init_detector, inference_detector
        mmdet_available = True
        print("✓ mmdet.apis导入成功")
    except Exception as e:
        print(f"✗ mmdet.apis导入失败: {e}")

    def parse_args():
        parser = argparse.ArgumentParser(description='Validation script for PyTorch and Jittor models')
        parser.add_argument('--config', default='configs/D2Det/D2Det_detection_r50_fpn_custom.py', 
                          help='config file path')
        parser.add_argument('--checkpoint', default='/home/w3k0lss/D2Det/work_dirs/d2det_pytorch/latest.pth', 
                          help='checkpoint file path')
        parser.add_argument('--work_dir', default='work_dirs/d2det_unified_val', 
                          help='the dir to save logs and results')
        parser.add_argument('--device', default='cpu', help='device to use (cpu, cuda, cuda:0)')
        parser.add_argument('--framework', default='auto', 
                          choices=['auto', 'pytorch', 'jittor'],
                          help='framework to use (default: auto detect from checkpoint)')
        parser.add_argument('--num_images', type=int, default=20, 
                          help='number of images to process (default: 20)')
        args = parser.parse_args()
        return args

    def detect_framework(checkpoint_path):
        if not osp.exists(checkpoint_path):
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint or 'model' in checkpoint:
                return 'pytorch'
        except:
            pass
        
        try:
            import jittor as jt
            checkpoint = jt.load(checkpoint_path)
            return 'jittor'
        except:
            pass
        
        return 'unknown'

    def load_pytorch_model(cfg, checkpoint_path, device):
        if not mmdet_available:
            print("✗ mmdet.apis不可用，无法加载PyTorch模型")
            return None
        
        try:
            model = init_detector(cfg, checkpoint_path, device=device)
            print("✓ PyTorch模型加载成功！")
            return model
        except Exception as e:
            print(f"✗ PyTorch模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def inference_pytorch(model, img_path):
        try:
            result = inference_detector(model, img_path)
            return result
        except Exception as e:
            print(f"    推理失败: {e}")
            return None

    def load_jittor_model(cfg, checkpoint_path, device):
        try:
            import jittor as jt
            if device.startswith('cuda'):
                jt.flags.use_cuda = 1
                print("  使用CUDA加速")
            else:
                jt.flags.use_cuda = 0
                print("  使用CPU")
            
            print("  注意: Jittor模型加载需要Jittor版本的D2Det实现")
            print("  这里是一个占位实现，需要根据实际Jittor代码调整")
            return None
        except Exception as e:
            print(f"✗ Jittor模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def inference_jittor(model, img_path):
        try:
            print("  Jittor推理功能待实现")
            return None
        except Exception as e:
            print(f"    推理失败: {e}")
            return None

    def main():
        args = parse_args()

        cfg = Config.fromfile(args.config)
        cfg.work_dir = args.work_dir

        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        print(f"\n配置文件: {args.config}")
        print(f"检查点: {args.checkpoint}")
        print(f"工作目录: {args.work_dir}")
        print(f"设备: {args.device}")
        print(f"框架: {args.framework}")

        framework = args.framework
        if framework == 'auto':
            framework = detect_framework(args.checkpoint)
            if framework is None:
                print("✗ 无法自动检测框架类型")
                print("  请使用 --framework 参数手动指定")
                return
            print(f"✓ 自动检测到框架: {framework}")

        print("\n加载COCO标注文件...")
        ann_file = cfg.data.val.ann_file
        with open(ann_file, 'r') as f:
            coco_ann = json.load(f)
        
        print(f"✓ 标注文件加载成功")
        print(f"  图像数量: {len(coco_ann['images'])}")
        print(f"  标注数量: {len(coco_ann['annotations'])}")
        print(f"  类别数量: {len(coco_ann['categories'])}")

        categories = {cat['id']: cat['name'] for cat in coco_ann['categories']}
        CLASSES = [cat['name'] for cat in coco_ann['categories']]

        vis_dir = osp.join(cfg.work_dir, 'visualizations')
        mmcv.mkdir_or_exist(vis_dir)

        model = None
        if framework == 'pytorch' and pytorch_available:
            print("\n加载PyTorch模型...")
            model = load_pytorch_model(cfg, args.checkpoint, args.device)
        elif framework == 'jittor':
            print("\n加载Jittor模型...")
            model = load_jittor_model(cfg, args.checkpoint, args.device)

        print(f"\n处理前{args.num_images}张图像...")
        img_prefix = osp.join(cfg.data.val.img_prefix, 'val2017')
        
        results = []
        for i, img_info in enumerate(coco_ann['images'][:args.num_images]):
            print(f"  处理图像 {i+1}/{args.num_images}: {img_info['file_name']}")
            
            img_path = osp.join(img_prefix, img_info['file_name'])
            if not osp.exists(img_path):
                print(f"    图像不存在，跳过")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"    无法读取图像，跳过")
                continue
            
            anns = [ann for ann in coco_ann['annotations'] if ann['image_id'] == img_info['id']]
            
            img_gt = img.copy()
            for ann in anns:
                bbox = ann['bbox']
                x, y, w, h = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                class_id = ann['category_id']
                class_name = categories.get(class_id, str(class_id))
                
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_gt, class_name, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            img_pred = img.copy()
            result = None
            if model is not None:
                if framework == 'pytorch':
                    result = inference_pytorch(model, img_path)
                elif framework == 'jittor':
                    result = inference_jittor(model, img_path)
                
                if result is not None:
                    for class_idx, bboxes in enumerate(result):
                        if class_idx >= len(CLASSES):
                            break
                        class_name = CLASSES[class_idx]
                        for bbox in bboxes:
                            x1, y1, x2, y2, score = bbox
                            if score > 0.3:
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                label = f"{class_name}: {score:.2f}"
                                cv2.putText(img_pred, label, (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    results.append(result)
            
            vis_gt_path = osp.join(vis_dir, f'gt_{i:03d}.jpg')
            cv2.imwrite(vis_gt_path, img_gt)
            
            if model is not None and result is not None:
                vis_pred_path = osp.join(vis_dir, f'pred_{i:03d}.jpg')
                cv2.imwrite(vis_pred_path, img_pred)
                
                vis_combined = np.hstack([img_gt, img_pred])
                vis_combined_path = osp.join(vis_dir, f'combined_{i:03d}.jpg')
                cv2.imwrite(vis_combined_path, vis_combined)

        print("\n生成验证报告...")
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': args.config,
            'checkpoint': args.checkpoint,
            'device': args.device,
            'framework': framework,
            'model_loaded': model is not None,
            'dataset_stats': {
                'num_images': len(coco_ann['images']),
                'num_annotations': len(coco_ann['annotations']),
                'num_categories': len(coco_ann['categories']),
                'categories': CLASSES
            },
            'num_images_processed': min(args.num_images, len(coco_ann['images'])),
            'note': '统一验证脚本，支持PyTorch和Jittor两种框架'
        }

        report_file = osp.join(cfg.work_dir, 'validation_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"✓ 验证报告已保存到: {report_file}")

        print("\n" + "="*60)
        print("验证完成！")
        print("="*60)
        print("\n生成的文件:")
        print(f"  1. 验证报告: {report_file}")
        print(f"  2. 可视化结果: {vis_dir}/")
        print(f"\n使用说明:")
        print(f"  - 自动检测框架: python validate_unified.py")
        print(f"  - 指定PyTorch: python validate_unified.py --framework pytorch")
        print(f"  - 指定Jittor: python validate_unified.py --framework jittor")
        print(f"  - 处理更多图像: python validate_unified.py --num_images 100")

    if __name__ == '__main__':
        main()

except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
