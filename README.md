# 🎹 AI-Powered Piano Trainer

This project uses a Raspberry Pi, USB webcam, and a YOLOv8 model to detect pressed piano keys in real time. It compares detected keys to a predefined melody (“Twinkle Twinkle Little Star”) and provides instant feedback using an LCD display, buzzer, and logging system.

Originally built as a school project for Creative Technologies & AI (CTAI) 2025 at Howest.

---

##  Features

-  Real-time piano key detection using YOLOv8 and webcam
-  LCD display via I2C using bitwise operations
-  Socket communication between Pi and laptop
-  Accuracy logging saved as CSV
-  Optional buzzer melody at end of song
-  Song preview using audio playback


---

## 🔧 Requirements

**Hardware:**
- Raspberry Pi 4 or 5
- USB webcam
- I2C 16x2 LCD display (with PCF8574 backpack)
- Buzzer (connected to GPIO18)
- Piano keyboard (physical, not MIDI)

**Python Libraries:**
- `opencv-python`
- `smbus`
- `socket` (built-in)
- `playsound`
- `ultralytics` (YOLOv8)

---

##  Demo Video

Here’s a short school showcase video that gives a quick look at the full setup and how it works:

 [Watch it on YouTube](https://youtu.be/3MhScdFX6Gk)

---

##  How It Works (Quick Overview)

1. The Pi captures live video frames from the webcam.
2. Frames are sent to the laptop via socket.
3. The laptop runs YOLOv8 model inference to detect which key is pressed.
4. The detected key is sent back to the Pi over a second socket.
5. The Pi compares it with the expected song sequence and:
   - Displays result on LCD
   - Logs data to CSV
   - Plays buzzer melody at end
   - Prompts user to replay

---


##  Questions or feedback?

Feel free to open an issue or leave a comment on the Instructables post:  
 [AI-Powered Piano Trainer on Instructables](https://www.instructables.com/AI-Powered-Piano-Trainer-Learn-Songs-With-Real-Tim/)

---

##  License Notice

This project was created as part of coursework at Howest University.  
Before assigning an open-source license, I’m checking with the institution's policy on student project ownership.  
For now, please feel free to explore the code, but do not reuse or redistribute it without permission.
