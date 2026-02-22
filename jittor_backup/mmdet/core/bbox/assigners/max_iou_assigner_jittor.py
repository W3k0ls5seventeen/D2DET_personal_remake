import jittor as jt

from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Var): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Var, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.shape[0], overlaps.shape[1]

        # 1. assign -1 by default
        assigned_gt_inds = jt.full((num_bboxes, ), -1, dtype=jt.int64)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = jt.zeros((num_bboxes, ), dtype=overlaps.dtype)
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = jt.zeros((num_bboxes, ), dtype=jt.int64)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # 处理Jittor的max函数返回值
        max_result = overlaps.max(dim=0)
        max_overlaps = max_result[0] if isinstance(max_result, tuple) else max_result
        # 处理Jittor的argmax函数返回值
        argmax_result = overlaps.argmax(dim=0)
        argmax_overlaps = argmax_result[0] if isinstance(argmax_result, tuple) else argmax_result
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # 处理Jittor的max函数返回值
        gt_max_result = overlaps.max(dim=1)
        gt_max_overlaps = gt_max_result[0] if isinstance(gt_max_result, tuple) else gt_max_result
        # 处理Jittor的argmax函数返回值
        gt_argmax_result = overlaps.argmax(dim=1)
        gt_argmax_overlaps = gt_argmax_result[0] if isinstance(gt_argmax_result, tuple) else gt_argmax_result

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            neg_inds = (max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)
            if neg_inds.any():
                neg_inds_np = neg_inds.numpy()
                assigned_gt_inds_np = assigned_gt_inds.numpy()
                assigned_gt_inds_np[neg_inds_np] = 0
                assigned_gt_inds = jt.array(assigned_gt_inds_np)
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            neg_inds = (max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])
            if neg_inds.any():
                neg_inds_np = neg_inds.numpy()
                assigned_gt_inds_np = assigned_gt_inds.numpy()
                assigned_gt_inds_np[neg_inds_np] = 0
                assigned_gt_inds = jt.array(assigned_gt_inds_np)

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        # 使用where函数来处理布尔掩码索引
        if pos_inds.any():
            # 转换为numpy数组进行索引
            pos_inds_np = pos_inds.numpy()
            argmax_overlaps_np = argmax_overlaps.numpy()
            assigned_gt_inds_np = assigned_gt_inds.numpy()
            assigned_gt_inds_np[pos_inds_np] = argmax_overlaps_np[pos_inds_np] + 1
            assigned_gt_inds = jt.array(assigned_gt_inds_np)

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    # 转换为numpy数组进行索引
                    max_iou_inds_np = max_iou_inds.numpy()
                    assigned_gt_inds_np = assigned_gt_inds.numpy()
                    assigned_gt_inds_np[max_iou_inds_np] = i + 1
                    assigned_gt_inds = jt.array(assigned_gt_inds_np)
                else:
                    # 确保索引是整数
                    idx = int(gt_argmax_overlaps[i].numpy())
                    assigned_gt_inds_np = assigned_gt_inds.numpy()
                    assigned_gt_inds_np[idx] = i + 1
                    assigned_gt_inds = jt.array(assigned_gt_inds_np)

        if gt_labels is not None:
            assigned_labels = jt.zeros((num_bboxes, ), dtype=jt.int64)
            pos_inds = jt.nonzero(assigned_gt_inds > 0).squeeze(dim=1)
            if len(pos_inds) > 0:
                # 转换为numpy数组进行索引
                pos_inds_np = pos_inds.numpy()
                assigned_gt_inds_np = assigned_gt_inds.numpy()
                gt_labels_np = gt_labels.numpy()
                assigned_labels_np = assigned_labels.numpy()
                assigned_labels_np[pos_inds_np] = gt_labels_np[
                    assigned_gt_inds_np[pos_inds_np] - 1]
                assigned_labels = jt.array(assigned_labels_np)
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
