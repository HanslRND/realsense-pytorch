import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


logger = logging.getLogger(__name__)


def set_logging(rank=-1):
    logging.basicConfig(format="%(message)s", level=logging.INFO if rank in [-1, 0] else logging.WARN)


def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
    return new_size


def check_file(file):
    return file


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 4096
    max_det = 300
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        if not x.shape[0]:
            continue

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]

    return output


def increment_path(path, exist_ok=True, sep=""):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not Path(p).exists():
                return Path(p)
    return path
