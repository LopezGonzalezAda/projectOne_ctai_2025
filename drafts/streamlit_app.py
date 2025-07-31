import streamlit as st
import socket
import struct
import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient

# Streamlit UI setup
st.set_page_config(page_title="AI Piano Trainer", layout="centered")
st.title("🎹 AI-Powered Piano Trainer")
st.markdown("Live webcam with detected pressed piano keys")

# Roboflow model setup
rf = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="NW0OR7CAuzrGbEzhXpzF"  # 🔁 Replace with your Roboflow API key
)

# --- SOCKET SETUP ---

# 1. Listen for video stream from Pi (port 8000)
VIDEO_PORT = 8000
video_socket = socket.socket()
video_socket.bind(("0.0.0.0", VIDEO_PORT))
video_socket.listen(1)
print("[Server] Waiting for video stream...")
conn, _ = video_socket.accept()
conn_file = conn.makefile("rb")
print("[Server] Connected to video stream")

# 2. Connect to Pi to send predictions (port 9000)
RESULT_PORT = 9000
result_socket = socket.socket()
result_socket.connect(("192.168.168.167", RESULT_PORT))
print("[Server] Connected to Pi for predictions")

# UI placeholders
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

FRAME_WIDTH = 640
FRAME_SKIP = 5  # Inference every 5th frame

frame_count = 0
last_prediction = None
last_sent = None

try:
    while True:
        # Read frame size
        packed_size = conn_file.read(4)
        if not packed_size:
            break
        size = struct.unpack(">L", packed_size)[0]
        data = conn_file.read(size)

        # Decode frame from bytes
        img_array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        frame_count += 1
        display_frame = frame.copy()

        # Only run inference every N frames
        if frame_count % FRAME_SKIP == 0:
            result = rf.infer(pil_image, model_id="pressed_key_check/4")

            if "predictions" in result and result["predictions"]:
                pred = result["predictions"][0]
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                label = pred["class"]
                conf = pred["confidence"]
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if label != last_prediction:
                    last_prediction = label
                    prediction_placeholder.markdown(f"### 🔍 Detected: `{label}`")

                    if label != last_sent:
                        result_socket.sendall(label.encode())
                        last_sent = label
            else:
                prediction_placeholder.markdown("### 🔍 No key detected")
        else:
            # Show last known prediction
            if last_prediction:
                cv2.putText(display_frame, f"{last_prediction}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Detecting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize and show frame
        resized = cv2.resize(display_frame, (FRAME_WIDTH, int(display_frame.shape[0] * FRAME_WIDTH / display_frame.shape[1])))
        frame_for_ui = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        frame_placeholder.image(frame_for_ui, use_column_width=True)

except Exception as e:
    st.error(f"❌ Error: {e}")
finally:
    conn_file.close()
    conn.close()
    video_socket.close()
    result_socket.close()
