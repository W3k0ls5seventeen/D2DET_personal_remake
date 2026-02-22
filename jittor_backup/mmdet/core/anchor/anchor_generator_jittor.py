import jittor as jt
import numpy as np


class AnchorGenerator(object):
    """
    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2))
        >>> print(all_anchors)
        [[ 0.,  0.,  8.,  8.],
         [16.,  0., 24.,  8.],
         [ 0., 16.,  8., 24.],
         [16., 16., 24., 24.]]
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = jt.array(scales)
        self.ratios = jt.array(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.shape[0]

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = jt.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).reshape(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).reshape(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).reshape(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).reshape(-1)

        # yapf: disable
        base_anchors = jt.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        # 使用更兼容Jittor的方式实现meshgrid
        xx = x.repeat(len(y))
        yy = y.unsqueeze(1).repeat(1, len(x)).reshape(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device=None):
        base_anchors = self.base_anchors
        num_base_anchors = base_anchors.shape[0]

        feat_h, feat_w = featmap_size
        
        # 确保feat_h和feat_w是整数
        feat_h = int(feat_h)
        feat_w = int(feat_w)
        
        # 使用NumPy广播的方式生成锚框，提高效率
        # 生成偏移量数组
        shifts_x = np.arange(0, feat_w * stride, stride, dtype=np.float32)
        shifts_y = np.arange(0, feat_h * stride, stride, dtype=np.float32)
        shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=1)
        
        # 为每个偏移量添加base anchors
        base_anchors_np = base_anchors.numpy()
        all_anchors = (shifts[:, None, :] + base_anchors_np[None, :, :]).reshape(-1, 4)
        
        # 转换为Jittor张量
        all_anchors = jt.array(all_anchors)
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device=None):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = jt.zeros(feat_w, dtype='uint8')
        valid_y = jt.zeros(feat_h, dtype='uint8')
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.shape[0],
                                   self.num_base_anchors).reshape(-1)
        return valid
