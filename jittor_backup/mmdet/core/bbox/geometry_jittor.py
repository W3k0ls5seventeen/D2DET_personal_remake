import jittor as jt
import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    # 确保输入是Jittor变量
    if isinstance(bboxes1, torch.Tensor):
        bboxes1 = jt.array(bboxes1.detach().cpu().numpy())
    if isinstance(bboxes2, torch.Tensor):
        bboxes2 = jt.array(bboxes2.detach().cpu().numpy())
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Var): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Var): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Var): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = jt.float32([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = jt.float32([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        Var([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = jt.float32([])
        >>> nonempty = jt.float32([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return jt.zeros((rows, 1), dtype=bboxes1.dtype) if is_aligned else jt.zeros((rows, cols), dtype=bboxes1.dtype)

    if is_aligned:
        lt = jt.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = jt.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        # 确保广播正确发生
        # 将bboxes2调整为[1, cols, 2]形状
        bboxes2_reshaped = bboxes2.reshape(1, cols, 4)
        # 将bboxes1调整为[rows, 1, 2]形状
        bboxes1_lt = bboxes1[:, :2].reshape(rows, 1, 2)
        bboxes1_rb = bboxes1[:, 2:].reshape(rows, 1, 2)
        # 计算交集的左上角和右下角
        lt = jt.maximum(bboxes1_lt, bboxes2_reshaped[:, :, :2])  # [rows, cols, 2]
        rb = jt.minimum(bboxes1_rb, bboxes2_reshaped[:, :, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            # 确保area1和area2的形状正确
            area1_reshaped = area1.reshape(rows, 1)
            area2_reshaped = area2.reshape(1, cols)
            ious = overlap / (area1_reshaped + area2_reshaped - overlap)
        else:
            ious = overlap / (area1.reshape(rows, 1))

    return ious
