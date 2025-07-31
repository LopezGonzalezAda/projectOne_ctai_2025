import socket
from LCD import LCD
import csv
from datetime import datetime
import os
import RPi.GPIO as GPIO
import time

# 🛠️ Buzzer configuration
BUZZER_PIN = 4  # Change this to your actual GPIO pin

def play_buzzer_melody():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)

    pwm = GPIO.PWM(BUZZER_PIN, 440)  # A4
    pwm.start(50)

    time.sleep(0.3)
    pwm.ChangeFrequency(660)  # E5
    time.sleep(0.3)
    pwm.ChangeFrequency(880)  # A5
    time.sleep(0.3)

    pwm.stop()  # ✅ Explicitly stop before cleanup
    del pwm      # ❌ Avoid leaving destructor to clean up
    GPIO.cleanup()


# 🎵 Expected song sequence
expected = ['c', 'c', 'g', 'g', 'a', 'a', 'g',
            'f', 'f', 'e', 'e', 'd', 'd', 'c']

def run_song_loop(connection):
    lcd = LCD()
    detected = []

    # 🗂️ Setup logging
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/song_log_{timestamp}.csv"
    logfile = open(log_path, "w", newline="")
    logwriter = csv.writer(logfile)
    logwriter.writerow(["Timestamp", "Expected", "Detected", "Match"])

    lcd.clear()
    lcd.send_string("Waiting for key...", LCD.LCD_LINE_1)

    # 🔄 Flush leftover socket data
    connection.setblocking(False)
    try:
        while connection.recv(1024):
            pass
    except:
        pass
    connection.setblocking(True)

    # 🎼 Start receiving predictions
    while len(detected) < len(expected):
        data = connection.recv(1024).decode().strip().lower()
        if data.endswith('_pressed'):
            note = data[0]
            expected_note = expected[len(detected)]
            match = note == expected_note

            # 📝 Log result
            logwriter.writerow([
                datetime.now().strftime("%H:%M:%S"),
                expected_note,
                note,
                "Yes" if match else "No"
            ])

            detected.append(note)
            print(f"[Socket] Received: {note}")
            lcd.clear()
            lcd.send_string("Detected key:", LCD.LCD_LINE_1)
            lcd.send_string(note.upper(), LCD.LCD_LINE_2)

    logfile.close()

    # 📊 Calculate and show accuracy
    correct = sum(1 for a, b in zip(detected, expected) if a == b)
    accuracy = (correct / len(expected)) * 100
    lcd.clear()
    lcd.send_string("Song complete!", LCD.LCD_LINE_1)
    lcd.send_string(f"Acc: {accuracy:.1f}%", LCD.LCD_LINE_2)

    print("🎵 Song finished!")
    print(f"Detected sequence: {detected}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"📁 Log saved to: {log_path}")

    # 🔔 Play buzzer melody to indicate end
    play_buzzer_melody()

# -------- 🚀 Main program --------

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9000))
server_socket.listen(1)

print("[Socket] Waiting for connection...")
connection, addr = server_socket.accept()
print(f"[Socket] Connected by {addr}")

try:
    while True:
        run_song_loop(connection)
        answer = input("▶️  Play again? (y/n): ").strip().lower()
        if answer != 'y':
            break
finally:
    connection.close()
    server_socket.close()
    print("🔚 Exiting.")
