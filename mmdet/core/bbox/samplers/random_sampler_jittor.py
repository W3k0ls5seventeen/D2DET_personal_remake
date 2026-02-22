import jittor as jt

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Var, the returned indices will be a Var;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Var | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Var or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_var = isinstance(gallery, jt.Var)
        if not is_var:
            gallery = jt.array(gallery, dtype=jt.int64)
        perm = jt.randperm(gallery.shape[0], dtype=jt.int64)[:num]
        rand_inds = gallery[perm]
        if not is_var:
            rand_inds = rand_inds.numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = jt.nonzero(assign_result.gt_inds > 0)
        if len(pos_inds) != 0:
            pos_inds = pos_inds.squeeze(1)
        if len(pos_inds) <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = jt.nonzero(assign_result.gt_inds == 0)
        if len(neg_inds) != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
