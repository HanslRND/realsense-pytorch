FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    udev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -v -E '^(torch|torchvision)' requirements.txt > /tmp/requirements-docker.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt pyrealsense2

COPY . .

CMD ["python", "main.py", "--yolov7-dir", ".", "--weights", "models/weight/yolov7.pt", "--device", "0"]


