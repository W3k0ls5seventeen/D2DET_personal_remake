import jittor as jt

from mmdet.utils import util_mixins


class AssignResult(util_mixins.NiceRepr):
    """
    Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment

        gt_inds (Var): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.

        max_overlaps (Var): the iou between the predicted box and its
            assigned truth box.

        labels (None | Var): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Example:
        >>> # An assign result between 4 predicted boxes and 9 true boxes
        >>> # where only two boxes were assigned.
        >>> num_gts = 9
        >>> max_overlaps = jt.array([0, .5, .9, 0])
        >>> gt_inds = jt.array([-1, 1, 2, 0], dtype=jt.int64)
        >>> labels = jt.array([0, 3, 4, 0], dtype=jt.int64)
        >>> self = AssignResult(num_gts, gt_inds, max_overlaps, labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(4,), max_overlaps.shape=(4,),
                      labels.shape=(4,))>
        >>> # Force addition of gt labels (when adding gt as proposals)
        >>> new_labels = jt.array([3, 4, 5], dtype=jt.int64)
        >>> self.add_gt_(new_labels)
        >>> print(str(self))  # xdoctest: +IGNORE_WANT
        <AssignResult(num_gts=9, gt_inds.shape=(7,), max_overlaps.shape=(7,),
                      labels.shape=(7,))>
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    @property
    def num_preds(self):
        """
        Return the number of predictions in this assignment
        """
        return len(self.gt_inds)

    @property
    def info(self):
        """
        Returns a dictionary of info about the object
        """
        return {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }

    def __nice__(self):
        """
        Create a "nice" summary string describing this assign result
        """
        parts = []
        parts.append('num_gts={!r}'.format(self.num_gts))
        if self.gt_inds is None:
            parts.append('gt_inds={!r}'.format(self.gt_inds))
        else:
            parts.append('gt_inds.shape={!r}'.format(
                tuple(self.gt_inds.shape)))
        if self.max_overlaps is None:
            parts.append('max_overlaps={!r}'.format(self.max_overlaps))
        else:
            parts.append('max_overlaps.shape={!r}'.format(
                tuple(self.max_overlaps.shape)))
        if self.labels is None:
            parts.append('labels={!r}'.format(self.labels))
        else:
            parts.append('labels.shape={!r}'.format(tuple(self.labels.shape)))
        return ', '.join(parts)

    @classmethod
    def random(cls, **kwargs):
        """
        Create random AssignResult for tests or debugging.

        Kwargs:
            num_preds: number of predicted boxes
            num_gts: number of true boxes
            p_ignore (float): probability of a predicted box assinged to an
                ignored truth
            p_assigned (float): probability of a predicted box not being
                assigned
            p_use_label (float | bool): with labels or not
            rng (None | int | numpy.random.RandomState): seed or state

        Returns:
            AssignResult :

        Example:
            >>> from mmdet.core.bbox.assigners.assign_result import *  # NOQA
            >>> self = AssignResult.random()
            >>> print(self.info)
        """
        from mmdet.core.bbox import demodata
        rng = demodata.ensure_rng(kwargs.get('rng', None))

        num_gts = kwargs.get('num_gts', None)
        num_preds = kwargs.get('num_preds', None)
        p_ignore = kwargs.get('p_ignore', 0.3)
        p_assigned = kwargs.get('p_assigned', 0.7)
        p_use_label = kwargs.get('p_use_label', 0.5)
        num_classes = kwargs.get('p_use_label', 3)

        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = jt.zeros(num_preds, dtype=jt.float32)
            gt_inds = jt.zeros(num_preds, dtype=jt.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = jt.zeros(num_preds, dtype=jt.int64)
            else:
                labels = None
        else:
            import numpy as np
            # Create an overlap for each predicted box
            max_overlaps = jt.array(rng.rand(num_preds))

            # Construct gt_inds for each predicted box
            is_assigned = jt.array(rng.rand(num_preds) < p_assigned)
            # maximum number of assignments constraints
            n_assigned = min(num_preds, min(num_gts, int(is_assigned.sum().numpy())))

            assigned_idxs = np.where(is_assigned.numpy())[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned_np = is_assigned.numpy()
            is_assigned_np[:] = 0
            is_assigned_np[assigned_idxs] = 1
            is_assigned = jt.array(is_assigned_np)

            is_ignore_np = (rng.rand(num_preds) < p_ignore) & is_assigned.numpy()
            is_ignore = jt.array(is_ignore_np)

            gt_inds = jt.zeros(num_preds, dtype=jt.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = jt.array(true_idxs)
            gt_inds_np = gt_inds.numpy()
            gt_inds_np[is_assigned.numpy()] = true_idxs[:n_assigned].numpy()
            gt_inds = jt.array(gt_inds_np)

            gt_inds = jt.array(
                rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds_np = gt_inds.numpy()
            gt_inds_np[is_ignore.numpy()] = -1
            gt_inds_np[~is_assigned.numpy()] = 0
            gt_inds = jt.array(gt_inds_np)
            max_overlaps_np = max_overlaps.numpy()
            max_overlaps_np[~is_assigned.numpy()] = 0
            max_overlaps = jt.array(max_overlaps_np)

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = jt.zeros(num_preds, dtype=jt.int64)
                else:
                    labels = jt.array(
                        rng.randint(1, num_classes + 1, size=num_preds))
                    labels_np = labels.numpy()
                    labels_np[~is_assigned.numpy()] = 0
                    labels = jt.array(labels_np)
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        self_inds = jt.arange(
            1, len(gt_labels) + 1, dtype=jt.int64)
        self.gt_inds = jt.concat([self_inds, self.gt_inds])

        self.max_overlaps = jt.concat(
            [jt.ones(len(gt_labels), dtype=self.max_overlaps.dtype), self.max_overlaps])

        if self.labels is not None:
            self.labels = jt.concat([gt_labels, self.labels])
