import gradio as gr
import socket
import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
import time
import threading

# Roboflow model setup
rf = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"  # 🔁 Replace with your actual key
)

# Socket setup to send predictions to the Pi
PI_IP = "192.168.168.167"
RESULT_PORT = 9000
result_socket = socket.socket()

try:
    result_socket.connect((PI_IP, RESULT_PORT))
    print(f"[Connected] Sending predictions to {PI_IP}:{RESULT_PORT}")
except Exception as e:
    print(f"[Error] Could not connect to Pi: {e}")
    result_socket = None

latest_sent = None
latest_label = "Loading..."
frame_lock = threading.Lock()
display_frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Webcam reader thread
def capture_loop():
    global display_frame, latest_sent, latest_label
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize((416, 416))

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

            # Convert to bbox
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Send to Pi if new
            if result_socket and label != latest_sent:
                try:
                    result_socket.sendall(label.encode())
                    latest_sent = label
                except Exception as e:
                    print(f"[Send error] {e}")

        with frame_lock:
            display_frame = frame.copy()
            latest_label = label

        time.sleep(0.1)  # adjust for speed vs load

# Gradio live update function
def get_frame():
    with frame_lock:
        frame_bgr = display_frame.copy()
        label = latest_label
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, f"🔍 {label}"

# Start webcam capture in a background thread
threading.Thread(target=capture_loop, daemon=True).start()

# Gradio app
gr.Interface(
    fn=get_frame,
    inputs=None,
    outputs=["image", "text"],
    title="🎹 AI-Powered Piano Trainer",
    live=True
).launch()
