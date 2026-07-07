from pathlib import Path
import time

import numpy as np
import torch

from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box


class TensorRTEngine:
    def __init__(self, engine_path: Path) -> None:
        try:
            import tensorrt as trt
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError("TensorRT runtime requires NVIDIA TensorRT Python and cuda-python packages") from exc

        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)
        with engine_path.open("rb") as file, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(file.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = self._find_io_names()
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if any(dim < 0 for dim in self.input_shape):
            raise RuntimeError("Dynamic TensorRT input shapes are not supported by this runner")
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        self.output = np.empty(self.output_shape, dtype=self.output_dtype)
        self.input_nbytes = int(np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize)
        self.output_nbytes = int(self.output.nbytes)
        self.stream = self._cuda_call(cudart.cudaStreamCreate())
        self.input_device = self._cuda_call(cudart.cudaMalloc(self.input_nbytes))
        self.output_device = self._cuda_call(cudart.cudaMalloc(self.output_nbytes))
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        self.context.set_tensor_address(self.output_name, int(self.output_device))

    def __del__(self) -> None:
        if hasattr(self, "cudart"):
            for attr in ("input_device", "output_device"):
                ptr = getattr(self, attr, None)
                if ptr:
                    self.cudart.cudaFree(ptr)
            stream = getattr(self, "stream", None)
            if stream:
                self.cudart.cudaStreamDestroy(stream)

    def _cuda_call(self, result):
        if not isinstance(result, tuple):
            raise RuntimeError(f"Unexpected CUDA result: {result}")
        error = result[0]
        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA call failed: {error}")
        return result[1] if len(result) > 1 else None

    def _find_io_names(self) -> tuple[str, str]:
        inputs = []
        outputs = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one TensorRT input and one output, got inputs={inputs}, outputs={outputs}")
        return inputs[0], outputs[0]

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        tensor = np.ascontiguousarray(tensor.astype(self.input_dtype, copy=False))
        if tuple(tensor.shape) != self.input_shape:
            raise ValueError(f"TensorRT input shape {tensor.shape} does not match engine shape {self.input_shape}")

        self._cuda_call(
            self.cudart.cudaMemcpyAsync(
                self.input_device,
                tensor.ctypes.data,
                self.input_nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
        )
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT execution failed")
        self._cuda_call(
            self.cudart.cudaMemcpyAsync(
                self.output.ctypes.data,
                self.output_device,
                self.output_nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        )
        self._cuda_call(self.cudart.cudaStreamSynchronize(self.stream))
        return self.output


class TensorRTYoloV7Detector:
    def __init__(
        self,
        engine: Path,
        weights: Path,
        img_size: int,
        conf_thres: float,
        iou_thres: float,
    ) -> None:
        if not engine.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine}")
        if not weights.exists():
            raise FileNotFoundError(f"YOLOv7 weights not found: {weights}")

        self.engine = TensorRTEngine(engine)
        self.torch = torch
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        model = checkpoint["ema" if checkpoint.get("ema") else "model"]
        self.names = model.module.names if hasattr(model, "module") else model.names
        self.stride = int(model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)
        input_height, input_width = self.engine.input_shape[-2:]
        if input_height != self.img_size or input_width != self.img_size:
            raise ValueError(
                f"Engine input is {input_width}x{input_height}, but --img-size resolved to {self.img_size}. "
                "Rebuild the engine or pass the matching --img-size."
            )
        self.colors = [[int(x) for x in np.random.randint(0, 255, 3)] for _ in self.names]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def annotate(self, frame: np.ndarray, depth_frame: np.ndarray, profile: bool = False) -> tuple[np.ndarray, int, dict[str, float]]:
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        start = time.perf_counter()
        img = letterbox(frame, self.img_size, auto=False, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        timings["preprocess_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        img = (img.astype(np.float32) / 255.0)[None]
        timings["tensor_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        pred_np = self.engine.infer(img)
        timings["inference_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        pred = torch.from_numpy(pred_np.astype(np.float32, copy=False))
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]
        timings["nms_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        annotated = frame.copy()
        count = 0
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords((self.img_size, self.img_size), pred[:, :4], annotated.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                count += 1
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1 = max(0, min(x1, depth_frame.shape[1]))
                x2 = max(0, min(x2, depth_frame.shape[1]))
                y1 = max(0, min(y1, depth_frame.shape[0]))
                y2 = max(0, min(y2, depth_frame.shape[0]))
                depth_roi = depth_frame[y1:y2, x1:x2]
                depth_value = np.median(depth_roi) if depth_roi.size else 0
                label = f"{self.names[int(cls)]} {conf:.2f} Depth: {depth_value / 1000:.2f}m"

                if label.startswith("person"):
                    plot_one_box(
                        xyxy,
                        annotated,
                        label=label,
                        color=self.colors[int(cls)],
                        line_thickness=2,
                    )
        timings["draw_ms"] = (time.perf_counter() - start) * 1000
        timings["detector_total_ms"] = (time.perf_counter() - total_start) * 1000
        return annotated, count, timings

