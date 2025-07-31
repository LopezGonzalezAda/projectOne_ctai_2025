import queue
import threading
import time
from datetime import datetime
from LCD import LCD
from ble_utils.bluetooth_uart_server import ble_gatt_uart_loop, get_ble_mac

# ------------- BLE + DISPLAY LOGIC -----------------

expected_sequence = ['c', 'c', 'g', 'g', 'a', 'a', 'g',
                     'f', 'f', 'e', 'e', 'd', 'd', 'c']
detected_sequence = []

rx_q = queue.Queue()
tx_q = queue.Queue()
evt_q = queue.Queue()

def now():
    return datetime.now().strftime("%H:%M:%S")

def init_ble():
    device_name = "PianoTrainer"
    print(f"[{now()}] [BLE] Starting advertising as: {device_name}")
    ble_thread = threading.Thread(
        target=ble_gatt_uart_loop,
        args=(rx_q, tx_q, device_name, evt_q),
        daemon=True
    )
    ble_thread.start()

def main():
    lcd = LCD()
    lcd.send_string("Waiting for keys", LCD.LCD_LINE_1)
    init_ble()

    try:
        while len(detected_sequence) < len(expected_sequence):
            try:
                key_bytes = rx_q.get(timeout=10)
                key = key_bytes.decode("utf-8").strip().lower()

                if key.endswith("_pressed"):
                    note = key[0]

                    # Only append if it's different from the last detected note
                    if len(detected_sequence) == 0 or detected_sequence[-1] != note:
                        detected_sequence.append(note)
                        lcd.clear()
                        lcd.send_string("Detected key:", LCD.LCD_LINE_1)
                        lcd.send_string(note.upper(), LCD.LCD_LINE_2)
                        print(f"[{now()}] [Sequence] {detected_sequence}")

            except queue.Empty:
                print(f"[{now()}] [Warning] No BLE input received")
                continue

        # 🧠 Compute accuracy
        correct = sum(1 for a, b in zip(detected_sequence, expected_sequence) if a == b)
        accuracy = (correct / len(expected_sequence)) * 100

        # 📺 Display result
        lcd.clear()
        lcd.send_string("Song complete!", LCD.LCD_LINE_1)
        lcd.send_string(f"Acc: {accuracy:.1f}%", LCD.LCD_LINE_2)
        print(f"[{now()}] [Result] Final sequence: {detected_sequence}")
        print(f"[{now()}] [Result] Accuracy: {accuracy:.1f}%")

    except KeyboardInterrupt:
        lcd.clear()
        lcd.send_string("Cancelled", LCD.LCD_LINE_1)
        print(f"[{now()}] [Stopped] Keyboard interrupt")

if __name__ == "__main__":
    main()
