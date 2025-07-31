from smbus import SMBus
import time


class LCD:

    I2C_ADDR = 0x27 
    LCD_CHR = 1      
    LCD_CMD = 0       
    LCD_LINE_1 = 0x80 
    LCD_LINE_2 = 0xC0 
    LCD_BACKLIGHT = 0x08  
    ENABLE = 0b00000100   
    E_DELAY = 0.0005    

    def __init__(self):
        self.bus = SMBus(1)
        self.lcd_init()

    def lcd_init(self):
        self.send_byte(0x33, self.LCD_CMD) 
        self.send_byte(0x32, self.LCD_CMD) 
        self.send_byte(0x06, self.LCD_CMD) 
        self.send_byte(0x0C, self.LCD_CMD) 
        self.send_byte(0x28, self.LCD_CMD) 
        self.send_byte(0x01, self.LCD_CMD) 
        time.sleep(0.05)

    def send_byte(self, bits, mode):
        bits_high = mode | (bits & 0xF0) | self.LCD_BACKLIGHT
        bits_low = mode | ((bits << 4) & 0xF0) | self.LCD_BACKLIGHT
        self.bus.write_byte(self.I2C_ADDR, bits_high)
        self.send_byte_with_e_toggle(bits_high)
        self.bus.write_byte(self.I2C_ADDR, bits_low)
        self.send_byte_with_e_toggle(bits_low)

    def send_byte_with_e_toggle(self, bits):
        time.sleep(self.E_DELAY)
        self.bus.write_byte(self.I2C_ADDR, (bits | self.ENABLE))
        time.sleep(self.E_DELAY)
        self.bus.write_byte(self.I2C_ADDR, (bits & ~self.ENABLE))
        time.sleep(self.E_DELAY)

    def send_instruction(self, instruction):
        self.send_byte(instruction, self.LCD_CMD)

    def send_character(self, char):
        self.send_byte(ord(char), self.LCD_CHR)

    def send_string(self, message, line=LCD_LINE_1):
        self.send_instruction(line)
        for char in message:
            self.send_character(char)

    def clear(self):
        self.send_instruction(0x01)

    def display_on(self):
        self.send_instruction(0x0C)

    def display_off(self):
        self.send_instruction(0x08)

    def cursor_on(self):
        self.send_instruction(0x0E)

    def cursor_off(self):
        self.send_instruction(0x0C)