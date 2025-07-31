import threading
import queue
import time
import smbus
import RPi.GPIO as GPIO

from ble_utils.bluetooth_uart_server import ble_gatt_uart_loop, get_ble_mac

# ----------------- LCD CONFIG --------------------
i2c = smbus.SMBus(1)
I2C_ADDR = 0x27
LCD_WIDTH = 16

LCD_CHR = 1
LCD_CMD = 0

LCD_LINE_1 = 0x80 | 0x00
LCD_LINE_2 = 0x80 | 0x40

LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100

E_PULSE = 0.0002
E_DELAY = 0.0002

# -------------- LCD FUNCTIONS --------------------

def lcd_send_byte_with_e_toggle(byte):
    time.sleep(E_DELAY)
    i2c.write_byte(I2C_ADDR, byte | ENABLE)
    time.sleep(E_PULSE)
    i2c.write_byte(I2C_ADDR, byte & ~ENABLE)
    time.sleep(E_DELAY)

def lcd_send_instruction(byte):
    lcd_set_data_bits(byte, LCD_CMD)

def lcd_send_character(byte):
    lcd_set_data_bits(byte, LCD_CHR)

def lcd_set_data_bits(byte, mode):
    MSNibble = byte & 0xF0
    LSNibble = (byte & 0x0F) << 4
    lcd_send_byte_with_e_toggle(MSNibble | mode | LCD_BACKLIGHT)
    lcd_send_byte_with_e_toggle(LSNibble | mode | LCD_BACKLIGHT)

def lcd_send_string(message, line=LCD_LINE_1):
    lcd_send_instruction(line)
    message = message.ljust(LCD_WIDTH)
    for char in message:
        lcd_send_character(ord(char))

def lcd_clear():
    lcd_send_instruction(0x01)
    time.sleep(0.05)
    lcd_send_instruction(0x02)
    time.sleep(0.05)

def lcd_init():
    lcd_send_byte_with_e_toggle(0x30)
    lcd_send_byte_with_e_toggle(0x30)
    lcd_send_byte_with_e_toggle(0x20)
    lcd_send_instruction(0x28)  # 4-bit mode, 2 lines, 5x8 font
    lcd_send_instruction(0x0C)  # Display ON, Cursor OFF, Blink OFF
    lcd_clear()

# -------------- BLE MAIN LOOP --------------------

def main():
    rx_q = queue.Queue()
    tx_q = queue.Queue()
    evt_q = queue.Queue()

    device_name = "piofada"
    ble_mac = get_ble_mac()

    lcd_init()
    lcd_send_string("BLE Device:")
    lcd_send_string(ble_mac, LCD_LINE_2)

    threading.Thread(target=ble_gatt_uart_loop, args=(rx_q, tx_q, device_name, evt_q), daemon=True).start()

    try:
        while True:
            try:
                incoming = rx_q.get_nowait()
                print(f"[BLE RX] {incoming}")

                lcd_clear()
                lcd_send_string("From BLE:")
                lcd_send_string(incoming[:16], LCD_LINE_2)  # limit to 16 chars

            except queue.Empty:
                pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        lcd_clear()
        lcd_send_string("Goodbye!")

if __name__ == '__main__':
    main()