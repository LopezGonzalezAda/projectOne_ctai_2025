import socket
import struct
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from datetime import datetime
from collections import deque
import time
import os
import threading
from playsound import playsound

from playsound import playsound
import threading
import os
import time

# Ask user whether to play the preview
answer = input("▶️  Do you want to listen to the song preview before starting? (y/n): ").strip().lower()

if answer == "y":
    audio_path = os.path.join(os.path.dirname(__file__), "assets", "twinkle_preview.mp3")
    threading.Thread(target=playsound, args=(audio_path,), daemon=True).start()
    print("🎶 Playing song preview...")
    time.sleep(6)  # Adjust this to match your song length







# 🔍 Roboflow Inference setup
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"
)
model_id = "pressed_key_check/4"
CONFIDENCE_THRESHOLD = 0.4

# 📡 Receive video from Pi
video_socket = socket.socket()
video_socket.bind(('0.0.0.0', 8000))
video_socket.listen(0)
video_conn, _ = video_socket.accept()
video_file = video_conn.makefile('rb')

print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to Pi video stream")

# 🎯 Send predictions to Pi
result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result_socket.connect(("192.168.168.167", 9000))  # Replace with your Pi's IP
print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to Pi result socket")

recent_keys = deque(maxlen=10)
frame_count = 0
last_predictions = []
last_time = time.time()

try:
    while True:
        # 🔄 Read frame size
        packed_size = video_file.read(4)
        if not packed_size:
            break
        size = struct.unpack(">L", packed_size)[0]

        # 📦 Read full image
        data = b''
        while len(data) < size:
            packet = video_file.read(size - len(data))
            if not packet:
                break
            data += packet

        # 🖼 Decode frame
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print("❌ Failed to decode frame")
            continue

        display_frame = frame.copy()
        frame_count += 1

        # 🧠 Run inference every 3 frames
        if frame_count % 3 == 0:
            try:
                response = client.infer(frame, model_id)
                last_predictions = response.get("predictions", [])
            except Exception as e:
                print("❌ Inference failed:", e)
                last_predictions = []

        # 🎯 Draw bounding boxes
        for pred in last_predictions:
            conf = pred["confidence"]
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            label = pred["class"]

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{label} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ✅ Send most confident label
        if last_predictions:
            top_pred = max(last_predictions, key=lambda p: p['confidence'])
            if top_pred["confidence"] >= CONFIDENCE_THRESHOLD:
                label = top_pred["class"]
                conf = top_pred["confidence"]
                already_sent = any(k == label and time.time() - t < 1.0 for k, t in recent_keys)
                if not already_sent:
                    result_socket.sendall(f"{label}\n".encode())
                    recent_keys.append((label, time.time()))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent: {label} ({conf:.2f})")

                cv2.putText(display_frame, f"Key: {label.upper()} ({conf:.2f})",
                            (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        # ⏱ FPS overlay
        curr_time = time.time()
        delta = curr_time - last_time
        fps = 1 / delta if delta > 0 else 0
        last_time = curr_time
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 🖼 Show live result
        cv2.imshow("Live Piano Detection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting viewer.")
            break

finally:
    video_file.close()
    video_conn.close()
    video_socket.close()
    result_socket.close()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Server shutdown")
