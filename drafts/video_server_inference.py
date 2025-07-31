# Laptop - video_server_inference.py
import socket
import struct
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from datetime import datetime
from collections import deque
import time

def now():
    return datetime.now().strftime("%H:%M:%S")

# 🔧 Setup Roboflow Inference
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"
)
model_id = "pressed_key_check/4"
CONFIDENCE_THRESHOLD = 0.5

# 🔧 Setup socket for receiving frames (from Pi)
video_socket = socket.socket()
video_socket.bind(('0.0.0.0', 8000))
video_socket.listen(0)
video_conn, _ = video_socket.accept()
video_file = video_conn.makefile('rb')

print(f"[{now()}] Connected to Pi video stream")

# 🔧 Setup socket for sending predictions (to Pi)
result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result_socket.connect(("192.168.168.167", 9000))  # Replace with Pi IP
print(f"[{now()}] Connected to Pi result socket")

# ⏱️ Store recently sent keys (with timestamps)
recent_keys = deque(maxlen=10)

try:
    while True:
        # 🔁 Read frame size
        packed_size = video_file.read(4)
        if not packed_size:
            break
        size = struct.unpack(">L", packed_size)[0]

        # 🔁 Read full frame data
        data = b''
        while len(data) < size:
            packet = video_file.read(size - len(data))
            if not packet:
                break
            data += packet

        # 🔁 Decode frame
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[{now()}] ❌ Failed to decode frame")
            continue

        # 📤 Inference
        response = client.infer(frame, model_id)
        predictions = response.get("predictions", [])

        for p in predictions:
            key = p["class"]
            conf = p["confidence"]

            if conf >= CONFIDENCE_THRESHOLD:
                already_sent = any(k == key and time.time() - t < 1.0 for k, t in recent_keys)
                if not already_sent:
                    result_socket.sendall(f"{key}\n".encode())
                    recent_keys.append((key, time.time()))
                    print(f"[{now()}] Sent: {key} ({conf:.2f})")

finally:
    video_file.close()
    video_conn.close()
    video_socket.close()
    result_socket.close()
    print(f"[{now()}] Server shutdown")
