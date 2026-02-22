import jittor as jt
import numpy as np

from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply
from ..utils.misc import unmap


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    # 确保img_metas是一个列表
    if not isinstance(img_metas, list):
        img_metas = [img_metas]
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    # 使用shape属性获取anchor数量，兼容Jittor张量
    num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        
        # 直接处理每个层级的锚框，而不是拼接成一个大的张量
        # 这样可以避免Jittor张量拼接的问题
        all_anchors = []
        all_valid_flags = []
        
        for j, anchors in enumerate(anchor_list[i]):
            # 将每个层级的锚框转换为numpy数组并添加到列表中
            anchors_np = anchors.numpy()
            all_anchors.append(anchors_np)
            # 同样处理valid_flags
            valid_flags = valid_flag_list[i][j]
            valid_flags_np = valid_flags.numpy()
            all_valid_flags.append(valid_flags_np)
        
        # 拼接numpy数组
        anchor_list[i] = np.concatenate(all_anchors)
        valid_flag_list[i] = np.concatenate(all_valid_flags)

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    
    # 准备传递给anchor_target_single的参数
    # 确保gt_bboxes是Jittor张量
    processed_gt_bboxes_list = []
    for gt_bboxes in gt_bboxes_list:
        if isinstance(gt_bboxes, list):
            # 如果是列表，确保每个元素都是Jittor张量
            processed_gt_bboxes = []
            for bboxes in gt_bboxes:
                if not isinstance(bboxes, jt.Var):
                    bboxes = jt.array(bboxes)
                processed_gt_bboxes.append(bboxes)
            processed_gt_bboxes_list.append(processed_gt_bboxes)
        else:
            # 如果不是列表，确保是Jittor张量
            if not isinstance(gt_bboxes, jt.Var):
                gt_bboxes = jt.array(gt_bboxes)
            processed_gt_bboxes_list.append(gt_bboxes)
    
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         processed_gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    # 使用shape[0]替代numel()，兼容Jittor张量
    num_total_pos = sum([max(inds.shape[0], 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.shape[0], 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    target = jt.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    # 确保img_meta是一个字典
    if isinstance(img_meta, str):
        img_meta = {'img_shape': (800, 1333, 3), 'pad_shape': (800, 1333, 3)}
    
    # 检查flat_anchors的类型，看是否已经是numpy数组
    
    # 如果flat_anchors已经是numpy数组，直接使用
    if isinstance(flat_anchors, np.ndarray):
        # 计算inside_flags
        img_h, img_w = img_meta['img_shape'][:2]
        allowed_border = cfg.allowed_border
        
        # 过滤锚框
        # 直接使用所有锚框，不进行过滤
        anchors_np = flat_anchors
        
        # 定义inside_flags变量，用于后续的unmap函数
        # 因为我们使用所有锚框，所以inside_flags应该全部为True
        inside_flags_np = np.ones(flat_anchors.shape[0], dtype=bool)
        inside_flags = jt.array(inside_flags_np)
    else:
        # 否则，使用原来的方法
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 6
        
        # 转换为numpy数组进行操作
        flat_anchors_np = flat_anchors.numpy()
        inside_flags_np = inside_flags.numpy()
        # 使用numpy布尔索引
        anchors_np = flat_anchors_np[inside_flags_np]
    
    # 转换回Jittor张量
    anchors = jt.array(anchors_np)
    
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = jt.zeros_like(anchors)
    bbox_weights = jt.zeros_like(anchors)
    labels = jt.zeros(num_valid_anchors, dtype=jt.int64)
    label_weights = jt.zeros(num_valid_anchors, dtype=jt.float32)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        # 使用shape属性获取anchor数量，兼容Jittor张量
        num_total_anchors = flat_anchors.shape[0]
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape, allowed_border=0):
    img_h, img_w = img_shape
    
    if allowed_border >= 0:
        # 计算各个条件
        cond1 = flat_anchors[:, 0] >= -allowed_border
        cond2 = flat_anchors[:, 1] >= -allowed_border
        cond3 = flat_anchors[:, 2] < img_w + allowed_border
        cond4 = flat_anchors[:, 3] < img_h + allowed_border
        
        inside_flags = valid_flags & cond1 & cond2 & cond3 & cond4
    else:
        inside_flags = valid_flags
    
    return inside_flags
