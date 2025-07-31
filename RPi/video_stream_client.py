# Pi - video_stream_client.py
import cv2
import socket
import struct
import time

# 📡 Connect to laptop server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.168.10', 8000))  # Replace with your laptop IP
conn = client_socket.makefile('wb')

# 🎥 Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise IOError("Cannot open camera")

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # 🌀 Optional: reduce resolution to boost FPS
        frame = cv2.resize(frame, (320, 320))

        # 📦 Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        data = jpeg.tobytes()

        # 📨 Send frame size and data
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)

        time.sleep(0.01)  # ~100 FPS cap, adjust as needed

finally:
    camera.release()
    conn.close()
    client_socket.close()
    print("📴 Stream stopped")
