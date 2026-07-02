import sys
import torch
from pathlib import Path
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box
from models.experimental import attempt_load
from utils.torch_utils import TracedModel, select_device

import numpy as np


class YoloV7Detector:
    def __init__(
        self,
        yolov7_dir: Path,
        weights: Path,
        device: str,
        img_size: int,
        conf_thres: float,
        iou_thres: float,
        trace: bool,
    ) -> None:
        if not yolov7_dir.exists():
            raise FileNotFoundError(f"YOLOv7 directory not found: {yolov7_dir}")
        if not weights.exists():
            raise FileNotFoundError(f"YOLOv7 weights not found: {weights}")

        sys.path.insert(0, str(yolov7_dir))


        original_torch_load = torch.load

        def trusted_yolov7_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        self.torch = torch
        self.device = select_device(device)
        self.half = self.device.type != "cpu"
        try:
            torch.load = trusted_yolov7_load
            self.model = attempt_load(str(weights), map_location=self.device)
        finally:
            torch.load = original_torch_load
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        if trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.colors = [[int(x) for x in np.random.randint(0, 255, 3)] for _ in self.names]
        if self.device.type != "cpu":
            warmup = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device)
            self.model(warmup.type_as(next(self.model.parameters())))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def annotate(self, frame: np.ndarray, depth_frame: np.ndarray) -> tuple[np.ndarray, int]:
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = self.torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with self.torch.no_grad():
            pred = self.model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        annotated = frame.copy()
        count = 0
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], annotated.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                count += 1
                depth_value = np.median(depth_frame[int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])])
                label = f"{self.names[int(cls)]} {conf:.2f} Depth: {depth_value/1000:.2f}m"

                if label.startswith("person"):
                    plot_one_box(
                        xyxy,
                        annotated,
                        label=label,
                        color=self.colors[int(cls)],
                        line_thickness=2,
                    )

        return annotated, count

