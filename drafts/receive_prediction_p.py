import socket
import os
import csv
import time
import threading
from datetime import datetime
from LCD import LCD
import RPi.GPIO as GPIO

# GPIO setup
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# Constants
HOST = '0.0.0.0'
PORT = 9000
EXPECTED = ['c', 'c', 'g', 'g', 'a', 'a', 'g',
            'f', 'f', 'e', 'e', 'd', 'd', 'c']
TEMPO_INTERVAL = 0.8  # seconds per beat

latest_prediction = None
lock = threading.Lock()

def prediction_listener(conn):
    global latest_prediction
    while True:
        try:
            data = conn.recv(1024).decode().strip().lower()
            if data:
                note = data.replace("_pressed", "")
                with lock:
                    latest_prediction = note
        except:
            break

def run_song_loop(connection):
    global latest_prediction

    lcd = LCD()
    detected = []
    beat = 0

    # Logging setup
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/song_log_{timestamp}.csv"
    logfile = open(log_path, "w", newline="")
    logwriter = csv.writer(logfile)
    logwriter.writerow(["Timestamp", "Expected", "Detected", "Match"])

    lcd.clear()
    lcd.send_string("Playing song...", LCD.LCD_LINE_1)

    # Start listener thread
    listener_thread = threading.Thread(target=prediction_listener, args=(connection,))
    listener_thread.daemon = True
    listener_thread.start()

    while len(detected) < len(EXPECTED):
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(LED_PIN, GPIO.LOW)

        time.sleep(TEMPO_INTERVAL - 0.1)

        with lock:
            if latest_prediction:
                expected_note = EXPECTED[len(detected)]
                match = latest_prediction == expected_note

                now = datetime.now().strftime("%H:%M:%S")
                logwriter.writerow([now, expected_note, latest_prediction, match])
                lcd.send_string(f"{latest_prediction.upper()} == {expected_note.upper()}", LCD.LCD_LINE_1)
                lcd.send_string("✓" if match else "✗", LCD.LCD_LINE_2)

                detected.append(latest_prediction)
                latest_prediction = None  # Reset for next beat
            else:
                lcd.send_string("No input", LCD.LCD_LINE_2)

    # Accuracy
    correct = sum(1 for d, e in zip(detected, EXPECTED) if d == e)
    accuracy = round(correct / len(EXPECTED) * 100)
    lcd.clear()
    lcd.send_string("🎵 Song finished!", LCD.LCD_LINE_1)
    lcd.send_string(f"Accuracy: {accuracy}%", LCD.LCD_LINE_2)
    logwriter.writerow(["Final Accuracy", "", "", f"{accuracy}%"])

    logfile.close()
    connection.close()
    GPIO.cleanup()

# Main server loop
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[Socket] Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        print(f"[Socket] Connected by {addr}")
        run_song_loop(conn)

if __name__ == "__main__":
    main()
