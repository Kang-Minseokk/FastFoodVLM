import os
import subprocess

# 1. YOLOv5 디렉토리로 이동
os.chdir("./yolov5-master")

# 2. export-coreml-nms.py 실행
subprocess.run([
    "python", "export-coreml-nms.py",
    "--img-size", "640",
    "--weights", "./runs/train/exp17/weights/best.pt",
    "--include", "coreml"
])
