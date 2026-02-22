from abc import ABCMeta, abstractmethod

import jittor as jt

from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Var): Boxes to be sampled from.
            gt_bboxes (Var): Ground truth bboxes.
            gt_labels (Var, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = jt.zeros((bboxes.shape[0], ), dtype=jt.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            # 确保所有输入都是Jittor变量
            if not isinstance(gt_bboxes, jt.Var):
                # 检查是否是PyTorch Tensor
                try:
                    import torch
                    if isinstance(gt_bboxes, torch.Tensor):
                        gt_bboxes = jt.array(gt_bboxes.detach().cpu().numpy())
                    else:
                        gt_bboxes = jt.array(gt_bboxes)
                except ImportError:
                    gt_bboxes = jt.array(gt_bboxes)
            if not isinstance(bboxes, jt.Var):
                # 检查是否是PyTorch Tensor
                try:
                    import torch
                    if isinstance(bboxes, torch.Tensor):
                        bboxes = jt.array(bboxes.detach().cpu().numpy())
                    else:
                        bboxes = jt.array(bboxes)
                except ImportError:
                    bboxes = jt.array(bboxes)
            bboxes = jt.concat([gt_bboxes, bboxes], dim=0)
            # 确保gt_labels是Jittor变量
            if not isinstance(gt_labels, jt.Var):
                # 检查是否是PyTorch Tensor
                try:
                    import torch
                    if isinstance(gt_labels, torch.Tensor):
                        gt_labels = jt.array(gt_labels.detach().cpu().numpy())
                    else:
                        gt_labels = jt.array(gt_labels)
                except ImportError:
                    gt_labels = jt.array(gt_labels)
            assign_result.add_gt_(gt_labels)
            gt_ones = jt.ones(gt_bboxes.shape[0], dtype=jt.uint8)
            gt_flags = jt.concat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = jt.unique(pos_inds)
        num_sampled_pos = pos_inds.shape[0]
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = jt.unique(neg_inds)

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
