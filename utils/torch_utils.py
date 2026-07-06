import math
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = __import__("logging").getLogger(__name__)


def select_device(device="", batch_size=None):
    cpu = device.lower() == "cpu"
    if cpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = not cpu and torch.cuda.is_available()
    logger.info(f"torch {torch.__version__} {'CUDA' if cuda else 'CPU'}")
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True,
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


def model_info(model, verbose=False, img_size=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:
        h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (include and k not in include) or k.startswith("_") or k in exclude:
            continue
        setattr(a, k, v)


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        return


def revert_sync_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        module_output = BatchNormXd(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    return module_output


class TracedModel(nn.Module):
    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super().__init__()
        self.stride = model.stride
        self.names = model.names
        self.model = revert_sync_batchnorm(model)
        self.model.to("cpu")
        self.model.eval()
        self.detect_layer = self.model.model[-1]
        self.model.traced = True
        rand_example = torch.rand(1, 3, img_size, img_size)
        self.model = torch.jit.trace(self.model, rand_example, strict=False)
        self.model.to(device)
        self.detect_layer.to(device)

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out
