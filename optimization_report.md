# YOLOv7 RealSense ROS2 노드 전환 및 TensorRT 최적화 보고서

작성일: 2026-07-07

## 1. 목적

기존 RealSense SDK 직접 캡처 방식의 YOLOv7 추론 코드를 ROS2 이미지 토픽 기반 노드로 전환하고, Jetson Orin 환경에서 실시간 처리 성능을 개선하는 것이 목적이었다.

목표 성능은 `/camera/yolo/processed` 토픽 기준 20 FPS 이상이었다.

## 2. 최종 상태 요약

- 입력 토픽
  - color: `/camera/camera/color/image_raw`
  - aligned depth: `/camera/camera/aligned_depth_to_color/image_raw`
- 출력 토픽
  - processed: `/camera/yolo/processed`
- 입력 해상도
  - `896x504`
- 출력 방식
  - OpenCV GUI 표시 제거
  - ROS2 `sensor_msgs/msg/Image` 토픽 publish 방식 사용
- 현재 최적화 결과
  - PyTorch 실행: 약 5.5~6 FPS
  - `--img-size` 축소 후: 약 4 FPS 개선
  - TensorRT 적용 후: 약 16 FPS 안정 동작
- 현재 한계
  - 목표인 20 FPS 이상에는 아직 미달
  - TensorRT 적용 후에도 추가 최적화 여지가 남아 있음

## 3. 주요 변경 사항

### 3.1 ROS2 토픽 기반 노드 전환

RealSense SDK 직접 캡처 코드를 제거하고 `rclpy` 기반 ROS2 노드로 변경했다.

처리 흐름은 다음과 같다.

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
        -> ROS2 subscriber
        -> YOLOv7 추론
        -> bbox/depth annotation
        -> /camera/yolo/processed publish
```

### 3.2 OpenCV GUI 제거

OpenCV 창 표시 방식은 Jetson 컨테이너 환경에서 검은 화면/멈춤 문제가 있었다. 따라서 `cv2.imshow()` 기반 표시를 제거하고, 처리된 이미지를 ROS2 토픽으로 publish하도록 변경했다.

### 3.3 cv_bridge 제거

NumPy 1.x 요구와 의존성 축소를 위해 `cv_bridge`를 제거하고, `sensor_msgs/msg/Image`를 직접 NumPy 배열로 변환하는 로직을 사용했다.

지원 encoding:

- `bgr8`
- `rgb8`
- `mono8`
- `8UC1`
- `16UC1`
- `mono16`
- `32FC1`

### 3.4 최신 프레임 처리 구조

초기 구조에서는 color callback 안에서 바로 YOLO 추론을 수행했다. 추론 시간이 길면 ROS callback 처리가 밀릴 수 있으므로, callback은 최신 frame만 저장하고 별도 worker thread가 최신 frame만 처리하도록 변경했다.

효과:

- 오래된 frame backlog 방지
- 항상 가장 최근 color/depth frame 기준 처리
- ROS callback blocking 완화

## 4. 컨테이너 및 ROS2 통신 이슈

### 4.1 host topic은 보이나 echo가 안 되는 문제

컨테이너에서 `ros2 topic list/info`는 보이지만 `ros2 topic echo`가 되지 않는 문제가 있었다. 원인은 Docker 네트워크와 DDS data plane/shared memory 경로 문제로 판단했다.

대응:

- `--net=host` 사용 확인
- 필요 시 `--ipc=host` 사용
- Fast DDS shared memory 우회 테스트로 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` 사용 가능성 확인

### 4.2 ROS Humble/Jazzy 혼용 문제

NVIDIA PyTorch iGPU 이미지가 Ubuntu 24.04 기반이라 `ros-humble-desktop` 설치가 되지 않았다. ROS Humble은 Ubuntu 22.04 대상이고, Ubuntu 24.04는 ROS Jazzy 대상이다.

따라서 컨테이너에는 Jazzy를 설치했다. 다만 host Humble과 container Jazzy를 혼용하면 다음과 같은 문제가 발생할 수 있었다.

```text
ResponseError("unknown tag 'rclpy.type_hash.TypeHash'")
```

이 문제는 host Humble CLI가 container Jazzy 쪽 타입 정보를 처리하면서 생기는 호환성 문제로 판단했다.

대응:

- container 내부 Jazzy CLI로 topic 확인
- RViz도 가능하면 같은 ROS 배포판 환경에서 확인

공식 참고:

- ROS Humble release: https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html
- ROS Jazzy release: https://docs.ros.org/en/jazzy/Releases/Release-Jazzy-Jalisco.html

## 5. PyTorch/CUDA 환경 이슈

### 5.1 CUDA driver mismatch

초기에는 PyTorch가 CUDA를 잡지 못했다.

증상:

```text
torch.cuda.is_available() -> False
The NVIDIA driver on your system is too old
```

원인은 설치된 PyTorch CUDA build와 host NVIDIA driver/JetPack 조합이 맞지 않는 것이었다.

### 5.2 Orin sm_87 미지원 PyTorch wheel 문제

이후 CUDA는 잡혔지만 다음 에러가 발생했다.

```text
CUDA error: no kernel image is available for execution on the device
```

확인 결과:

```text
torch.__version__ = 2.11.0+cu126
torch.version.cuda = 12.6
torch.cuda.get_arch_list() = ['sm_80', 'sm_90']
```

Jetson Orin은 `sm_87`이 필요하므로, 해당 PyTorch wheel은 Orin에서 실제 CUDA kernel 실행이 불가능했다.

대응:

- NVIDIA Jetson용 PyTorch/iGPU 이미지 사용
- 일반 PyPI CUDA wheel 대신 Jetson/Orin에 맞는 wheel 또는 컨테이너 사용

공식 참고:

- NVIDIA PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- NVIDIA CUDA GPU compute capability: https://developer.nvidia.com/cuda-gpus

## 6. 성능 프로파일링 결과

프로파일링 옵션을 추가했다.

```bash
python3 main.py --profile-every 30
```

출력 예:

```text
profile total=...ms convert=...ms pre=...ms tensor=...ms infer=...ms nms=...ms draw=...ms publish=...ms
```

측정 항목:

- `convert`: ROS Image -> OpenCV/Numpy 변환
- `pre`: letterbox resize/padding
- `tensor`: NumPy -> tensor 또는 TensorRT input 변환
- `infer`: 모델 forward/TensorRT engine inference
- `nms`: non-max suppression
- `draw`: bbox/depth annotation
- `publish`: processed image publish

PyTorch 기준 주요 결과:

```text
infer = 150~170ms
total = 160~180ms
```

해석:

- 전체 시간 대부분이 모델 forward에서 사용됨
- ROS publish, 변환, drawing보다 모델 추론이 주 병목
- 주변 코드 최적화만으로 20 FPS 달성은 어려움

PyTorch CUDA timing은 비동기 실행 때문에 단계별 정확한 측정 시 `torch.cuda.synchronize()`가 필요하다. 현재 코드는 `--profile-every`가 켜진 경우에만 동기화한다.

공식 참고:

- PyTorch CUDA semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- `torch.cuda.synchronize`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html

## 7. img-size 조정 결과

`--img-size`를 낮추면 모델 입력 해상도가 줄어 추론량이 감소했다.

결과:

- 기존 PyTorch 기준 약 5.5~6 FPS
- 입력 크기 축소 후 약 4 FPS 개선

해석:

- 입력 크기 축소는 즉시 효과가 있음
- 단, 탐지 정확도와 작은 객체 검출 성능 저하 가능성이 있음

## 8. TensorRT 적용

### 8.1 적용 방식

기존 PyTorch backend는 유지하고, `--engine` 인자가 있을 때만 TensorRT engine을 사용하는 선택형 backend를 추가했다.

변환 흐름:

```text
yolov7.pt
  -> ONNX export
  -> trtexec로 TensorRT engine 생성
  -> main.py에서 --engine으로 TensorRT runtime 사용
```

engine 생성 명령:

```bash
python3 export_tensorrt.py \
  --weights models/weight/yolov7.pt \
  --img-size 416 \
  --engine models/weight/yolov7.416.engine
```

실행 명령:

```bash
python3 main.py \
  --img-size 416 \
  --engine models/weight/yolov7.416.engine \
  --profile-every 30
```

TensorRT 관련 공식 참고:

- TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
- trtexec command-line tools: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html
- TensorRT Python API: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html

### 8.2 cuda-python 의존성

TensorRT Python runtime에서 CUDA memory copy/stream 실행을 위해 `cuda-python`이 필요했다.

증상:

```text
No module named 'cuda'
```

대응:

```bash
python3 -m pip install --break-system-packages "cuda-python>=12,<13"
```

확인:

```bash
python3 -c "import tensorrt; from cuda import cudart; print(tensorrt.__version__); print(cudart.cudaRuntimeGetVersion())"
```

### 8.3 letterbox shape 문제

TensorRT engine은 고정 입력 shape `(1, 3, 416, 416)`인데, `letterbox()` 기본값 `auto=True` 때문에 실제 입력이 `(1, 3, 256, 416)`으로 만들어졌다.

증상:

```text
TensorRT input shape (1, 3, 256, 416) does not match engine shape (1, 3, 416, 416)
```

원인:

- `auto=True`는 정사각형까지 padding하지 않고 stride 배수까지만 padding함
- PyTorch 경로에서는 동적 shape처럼 처리 가능했지만 TensorRT 고정 engine과는 맞지 않음

대응:

```python
img = letterbox(frame, self.img_size, auto=False, stride=self.stride)[0]
```

결과:

- TensorRT 입력이 항상 `416x416`으로 고정됨
- 기존 `yolov7.416.engine` 재생성 없이 해결 가능

## 9. 현재 성능 결과

최종 확인된 성능:

```text
TensorRT 적용 후 약 16 FPS 안정 동작
```

목표:

```text
20 FPS 이상
```

현재 차이:

- 목표 대비 약 4 FPS 부족
- TensorRT 적용으로 PyTorch 대비 큰 개선은 확인됨
- 그러나 전체 pipeline 기준 추가 병목이 남아 있음

## 10. 남은 병목 후보

TensorRT 적용 후에는 다음 항목을 다시 profile로 확인해야 한다.

```bash
python3 main.py --img-size 416 --engine models/weight/yolov7.416.engine --profile-every 30
```

확인 우선순위:

1. `infer`
   - 여전히 크면 TensorRT engine 자체가 병목
   - 더 작은 img-size 또는 INT8 검토

2. `nms`
   - TensorRT forward가 빨라진 뒤 Python/Torch NMS가 새 병목이 될 수 있음
   - EfficientNMS plugin 또는 TensorRT graph 내부 NMS 검토

3. `pre` / `tensor`
   - CPU resize, BGR->RGB, HWC->CHW, host->device copy 비용 확인
   - GPU preprocessing 또는 pinned memory 검토 가능

4. `publish`
   - ROS Image 직렬화/publish 비용 확인
   - QoS, 출력 해상도, publish rate 제한 검토 가능

5. `draw`
   - bbox drawing과 depth median 계산 비용 확인
   - detection 수가 많으면 depth ROI median이 비용이 될 수 있음

## 11. 20 FPS 이상을 위한 다음 작업안

### 11.1 가장 작은 추가 실험

```bash
python3 export_tensorrt.py \
  --weights models/weight/yolov7.pt \
  --img-size 384 \
  --engine models/weight/yolov7.384.engine

python3 main.py \
  --img-size 384 \
  --engine models/weight/yolov7.384.engine \
  --profile-every 30
```

기대:

- 416 대비 추론량 감소
- 20 FPS 도달 가능성 확인
- 정확도 저하 여부는 실제 카메라 환경에서 확인 필요

### 11.2 모델 축소

`yolov7.pt` full 모델 대신 `yolov7-tiny.pt` 계열로 변경하면 가장 큰 성능 향상이 예상된다.

장점:

- 20 FPS 이상 달성 가능성이 높음
- TensorRT와 조합 시 추가 개선 가능

단점:

- 정확도 저하 가능성
- 현재 weight와 class 구성 호환성 확인 필요

### 11.3 TensorRT NMS 통합

현재 구현은 모델 forward만 TensorRT로 돌리고 NMS는 기존 Python/Torch 경로를 사용한다.

NMS가 병목으로 확인되면 다음 단계는 TensorRT engine 내부에 NMS를 포함하는 것이다.

장점:

- CPU/PyTorch 후처리 비용 감소
- host/device 왕복 비용 감소 가능

단점:

- ONNX graph 수정 또는 TensorRT plugin 사용 필요
- 구현 복잡도 증가

### 11.4 INT8 적용

FP16 TensorRT보다 더 빠른 성능이 필요하면 INT8 quantization을 검토할 수 있다.

장점:

- Jetson Orin에서 성능 향상 가능

단점:

- calibration dataset 필요
- 정확도 손실 검증 필요
- 구현/검증 비용 증가

## 12. 결론

현재 병목은 초기 PyTorch 기준으로 모델 forward에 집중되어 있었고, TensorRT 적용으로 처리량은 약 16 FPS까지 개선되었다. 이는 PyTorch 5.5~6 FPS 대비 의미 있는 개선이다.

다만 목표인 20 FPS 이상에는 아직 도달하지 못했다. 다음 최적화는 코드 주변부보다 다음 순서가 타당하다.

1. TensorRT profile 로그로 `infer/nms/pre/publish` 재확인
2. `img-size 384` 또는 `320` TensorRT engine 실험
3. 목표 FPS가 우선이면 `yolov7-tiny` 계열 검토
4. 정확도를 유지해야 하면 TensorRT NMS 통합 또는 INT8 검토

현 단계에서는 TensorRT 적용이 효과적임은 확인되었고, 남은 4 FPS는 모델 크기, 입력 크기, NMS 후처리 중 어디를 희생하거나 최적화할지 결정해야 하는 구간이다.
