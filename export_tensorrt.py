#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

import torch

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device


class YoloV7OnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images):
        output = self.model(images, augment=False)
        return output[0] if isinstance(output, (tuple, list)) else output


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv7 weights to ONNX and TensorRT engine.")
    parser.add_argument("--weights", type=Path, default=Path("models/weight/yolov7.pt"))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workspace", type=int, default=2048, help="TensorRT workspace MiB")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def export_onnx(weights: Path, onnx_path: Path, img_size: int, device: str, opset: int) -> int:
    device_obj = select_device(device)
    original_torch_load = torch.load

    def trusted_yolov7_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = trusted_yolov7_load
        model = attempt_load(str(weights), map_location=device_obj).eval()
    finally:
        torch.load = original_torch_load
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    wrapper = YoloV7OnnxWrapper(model).to(device_obj).eval()
    dummy = torch.zeros(1, 3, img_size, img_size, device=device_obj)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
    )
    return img_size


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool, workspace: int) -> None:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        raise FileNotFoundError("trtexec not found. Install TensorRT tools or add trtexec to PATH.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace}",
    ]
    if fp16:
        command.append("--fp16")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    onnx_path = args.onnx or args.weights.with_suffix(f".{args.img_size}.onnx")
    engine_path = args.engine or args.weights.with_suffix(f".{args.img_size}.engine")
    img_size = export_onnx(args.weights, onnx_path, args.img_size, args.device, args.opset)
    build_engine(onnx_path, engine_path, fp16=not args.no_fp16, workspace=args.workspace)
    print(f"Exported TensorRT engine: {engine_path} (img-size={img_size})")


if __name__ == "__main__":
    main()

