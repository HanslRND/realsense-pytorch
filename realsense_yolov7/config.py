from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path


DEFAULT_YOLOV7_DIR = Path.home() / "Documents" / "GitHub" / "yolov7"


@dataclass(frozen=True)
class DemoConfig:
    yolov7_dir: Path
    weights: Path
    device: str
    img_size: int
    conf_thres: float
    iou_thres: float
    width: int
    height: int
    fps: int
    no_trace: bool


def parse_args() -> DemoConfig:
    parser = ArgumentParser(description="Run YOLOv7 detection on a RealSense color stream.")
    parser.add_argument("--yolov7-dir", type=Path, default=DEFAULT_YOLOV7_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_YOLOV7_DIR / "yolov7.pt")
    parser.add_argument("--device", default="", help="cuda device id like 0, or cpu")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.75)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-trace", action="store_true")
    return DemoConfig(**vars(parser.parse_args()))
