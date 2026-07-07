from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path


def default_yolov7_dir() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("realsense_yolov7"))
    except Exception:
        return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DemoConfig:
    yolov7_dir: Path
    weights: Path
    engine: Path | None
    device: str
    img_size: int
    conf_thres: float
    iou_thres: float
    color_topic: str
    depth_topic: str
    processed_topic: str
    expected_width: int
    expected_height: int
    no_trace: bool
    profile_every: int


def parse_args() -> DemoConfig:
    yolov7_dir = default_yolov7_dir()
    parser = ArgumentParser(description="Run YOLOv7 detection on ROS2 RealSense image topics.")
    parser.add_argument("--yolov7-dir", type=Path, default=yolov7_dir)
    parser.add_argument("--weights", type=Path, default=yolov7_dir / "models" / "weight" / "yolov7.pt")
    parser.add_argument("--engine", type=Path, default=None, help="TensorRT engine path. Uses PyTorch when omitted.")
    parser.add_argument("--device", default="", help="cuda device id like 0, or cpu")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.75)
    parser.add_argument("--iou-thres", type=float, default=0.65)
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--processed-topic", default="/camera/yolo/processed")
    parser.add_argument("--expected-width", type=int, default=896)
    parser.add_argument("--expected-height", type=int, default=504)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--profile-every", type=int, default=0, help="log per-stage timing every N processed frames")
    return DemoConfig(**vars(parser.parse_args()))
