import gradio as gr
import socket
import struct
import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
import threading
import time

# Roboflow model setup
rf = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"  # 🔁 Replace with your actual API key
)

# Receive video stream from Pi
VIDEO_PORT = 8000
video_socket = socket.socket()
video_socket.bind(("0.0.0.0", VIDEO_PORT))
video_socket.listen(1)
print("[Gradio] Waiting for video stream from Pi...")
conn, _ = video_socket.accept()
conn_file = conn.makefile("rb")
print("[Gradio] Connected to Pi video stream")

# Send predictions to Pi
RESULT_PORT = 9000
PI_IP = "192.168.168.167"
result_socket = socket.socket()
try:
    result_socket.connect((PI_IP, RESULT_PORT))
    print(f"[Gradio] Connected to Pi at {PI_IP}:{RESULT_PORT}")
except Exception as e:
    print(f"[Error] Cannot connect to Pi for predictions: {e}")
    result_socket = None

latest_sent = None
latest_label = "Waiting..."
display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame_lock = threading.Lock()

# Receive and process frames in background
def frame_receiver():
    global display_frame, latest_label, latest_sent

    while True:
        try:
            packed_size = conn_file.read(4)
            if not packed_size:
                continue
            size = struct.unpack(">L", packed_size)[0]
            data = conn_file.read(size)

            # Decode image
            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Resize for inference
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).resize((416, 416))

            # Inference
            try:
                result = rf.infer(pil_image, model_id="pressed_key_check/4")
            except Exception as e:
                print(f"[Inference error] {e}")
                continue

            label = "No key detected"

            if "predictions" in result and result["predictions"]:
                pred = result["predictions"][0]
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                label = pred["class"]
                conf = pred["confidence"]

                # Bounding box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Send to Pi
                if result_socket and label != latest_sent:
                    try:
                        result_socket.sendall(label.encode())
                        latest_sent = label
                    except Exception as e:
                        print(f"[Send error] {e}")

            with frame_lock:
                display_frame = frame.copy()
                latest_label = label

        except Exception as e:
            print(f"[Frame receive error] {e}")
            continue

# Launch receiver thread
threading.Thread(target=frame_receiver, daemon=True).start()

# Gradio display function
def get_latest():
    with frame_lock:
        frame = display_frame.copy()
        label = latest_label
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f"🔍 {label}"

# Gradio app
gr.Interface(
    fn=get_latest,
    inputs=None,
    outputs=["image", "text"],
    title="🎹 AI-Powered Piano Trainer",
    live=True
).launch()
