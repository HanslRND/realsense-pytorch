import torch
import torch.nn as nn


class SigmoidBin(nn.Module):
    def __init__(
        self,
        bin_count=10,
        min=0.0,
        max=1.0,
        reg_scale=2.0,
        use_loss_regression=True,
        use_fw_regression=True,
        BCE_weight=1.0,
        smooth_eps=0.0,
    ):
        super().__init__()
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.reg_scale = reg_scale
        self.use_fw_regression = use_fw_regression
        start = min + (self.scale / 2.0) / self.bin_count
        end = max - (self.scale / 2.0) / self.bin_count
        self.step = self.scale / self.bin_count
        self.register_buffer("bins", torch.arange(start, end + 0.0001, self.step).float())

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length
        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1 : 1 + self.bin_count]
        _, bin_idx = torch.max(pred_bin, dim=-1)
        result = pred_reg + self.bins[bin_idx] if self.use_fw_regression else self.bins[bin_idx]
        return result.clamp(min=self.min, max=self.max)
