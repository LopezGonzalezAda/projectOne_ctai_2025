import subprocess
import time

print("📡 Starting receive_prediction.py...")
recv = subprocess.Popen(["python3", "RPi/receive_prediction.py"])

time.sleep(2)

print("🎥 Starting video_stream_client.py...")
stream = subprocess.Popen(["python3", "RPi/video_stream_client.py"])

try:
    recv.wait()
    stream.wait()
except KeyboardInterrupt:
    recv.terminate()
    stream.terminate()
