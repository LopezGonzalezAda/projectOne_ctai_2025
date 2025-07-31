import requests
import numpy as np
import cv2
import time
from inference_sdk import InferenceHTTPClient
from datetime import datetime
from collections import deque
import socket

# --- CONFIG ---
SNAPSHOT_URL = "http://192.168.168.167:8080/?action=snapshot"  # Replace with your Pi IP
PI_RESULT_SOCKET = ("192.168.168.167", 9000)  # Pi's IP and port to send detected keys
MODEL_ID = "pressed_key_check/4"
CONFIDENCE_THRESHOLD = 0.4
SEND_COOLDOWN = 1.0  # seconds between sending same key

# Roboflow model setup
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"
)

# Setup socket to send detected keys to Pi
result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result_socket.connect(PI_RESULT_SOCKET)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to Pi result socket")

recent_keys = deque(maxlen=10)
last_time = time.time()

try:
    while True:
        # Capture snapshot from MJPEG stream
        try:
            response = requests.get(SNAPSHOT_URL, stream=True, timeout=2)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            print("⚠️ Failed to grab frame:", e)
            continue

        display_frame = frame.copy()

        # Run inference
        try:
            result = client.infer(frame, model_id=MODEL_ID)
            predictions = result.get("predictions", [])
        except Exception as e:
            print("❌ Inference failed:", e)
            predictions = []

        # Draw predictions
        for pred in predictions:
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

        # Send top prediction
        if predictions:
            top_pred = max(predictions, key=lambda p: p["confidence"])
            key = top_pred["class"]
            conf = top_pred["confidence"]

            if conf >= CONFIDENCE_THRESHOLD:
                already_sent = any(k == key and time.time() - t < SEND_COOLDOWN for k, t in recent_keys)
                if not already_sent:
                    result_socket.sendall(f"{key}\n".encode())
                    recent_keys.append((key, time.time()))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent: {key} ({conf:.2f})")

                cv2.putText(display_frame, f"Key: {key.upper()} ({conf:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        # Show frame with overlays
        delta = time.time() - last_time
        fps = 1 / delta if delta > 0 else 0
        last_time = time.time()
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live Piano Detection (MJPEG)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting viewer.")
            break

finally:
    result_socket.close()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Server shutdown")
