# Project One: Kickoff

## Step 0: Backup SD card & update RPi firmware

Create a backup of your microSD card using win32DiskImager or the Disks tool in OSX.

Also it might be needed to update your RPi firmware. You can write a special image to do this to your microSD card.

First select your device  
Next, for Operating System, select Misc utility images, Bootloader, SD Card Boot

This tool wil blink solid green while it's working. Once its **done** it will blink the green LED rapidly.

## Step 1: Prepare Raspberry Pi

### Downloading the image ⏳

- Download _[P1_2025_CTAI_v1.1.zip](https://studenthowest-my.sharepoint.com/:u:/g/personal/pieter-jan_beeckman_howest_be/EYasNwH1BmNMjzh4tR6ui50B6Metgzrj5v_S2MVlh2hCeA?e=xxspv3)_ to your local computer.

### Restoring the image ⏳

- ⚠ Unzip the file ⚠
- Place the file on an SD card of at least 16GB using Win32 Imager or Balena Etcher.
  If you want to use **The Raspberry Pi Imager** tool, **make sure you don't apply the OS customisation settings**
- After the image is written, you can remove the SD card and insert it into your Pi.

### Connecting the Pi

- Boot your Pi.
- Connect the Pi to your computer with a network cable and establish an SSH connection in _Putty_ or _Terminal_ to 192.168.168.**167** with username **_student_** and password **_P@ssw0rd_**

> CAUTION: The image is created in QWERTY, if you connect directly via keyboard  
> If logging in does not work, try in AZERTY _P2sszàrd_

- _After first login you have to set a new password for your (student) user_
- `ssh student@192.168.168.167` ; enter current password: _P@ssw0rd_ ; enter and confirm new password
- You get kicked from ssh, log in again
- ⚠ REBOOT your RPi (Because now VNC is not working yet)  
  `sudo reboot`

---

> **SSH**, **VNC** and **I²C** are already activated

---

## Step 2: Connect via VSCode, run and extend the kickoff demo project

0. **Open Visual Studio Code (VSCode) and Connect to the Raspberry Pi:**

   - Launch VSCode.
   - Use the RemoteSSH extension to connect to your Raspberry Pi.
   - Enter the connection string `ssh student@192.168.168.167 -A`.
   - Password: ~~`P@ssw0rd`~~ `yourNewlyCreatedPassword`
   - Operating System: `Linux`

1. **Open folder _/home/student/kickoff-demo_**

2. **Change _DEVICE_NAME_ on line 20**

3. **Run the _app.py_ to verify Bluetooth is working**

> no venv is needed, everything is installed system wide for this demo

> the classroom repo contains a similar demo _with_ a virtual environment

4. **Extend the _kickoff-demo/app.py_ with GPIO functionality**

- Button + Blue LED + RGB LED
- Button can toggle the Blue LED
- BLE client can send 3 commands: toggle, on and off  
  these commands control the Blue LED state
- BLE client gets notified if the Blue LED state changes by button press

You can follow the `# TODO:` in the code

Once this works, you can go on to the next step **_but you'll have to show this working at the end_** so keep these files

The idea is to forever keep this workspace / folder on the RPi so you can look back to a working example whenever you want or need.

**Keep the VSCode window (connected to the RPi) open for the next step**

## Step 3: Clone the Classroom Repo to the Raspberry Pi and Run the Bluetooth Server Code

0. **Follow [this link](https://classroom.github.com/a/5qgu4Pto) and accept the Repository invite**

> This step should already be done

1. **Clone the Classroom Repository:**

   - Open the Command Palette (`View > Command Palette`).
   - Type/search for clone
   - Select the `Git: Clone` option.
   - Copy the HTTPS URL of the classroom repository (avoid using the default SSH option).
   - Clone the repo into your home folder (`/home/student`)

2. **Open the Cloned Repository:**

   - Once cloned, open the folder in VSCode (in a new window)

3. **Set Up Python Virtual Environment:**

What is a Virtual Environment? See: https://datascientest.com/en/python-virtualenv-your-essential-guide-to-virtual-environments

- Open the Command Palette (`View > Command Palette`).
- Search for `Python: Create Environment`.
- Select `Venv` and choose any Python 3.11 interpreter.
- When prompted, check the `RPi/requirements.txt` and confirm by pressing OK.

5. **Modify the Bluetooth Server Code:**

   - Open `RPi/app.py` in VSCode.
   - Replace `"device_name"` on line 18 with a unique name for your device.

6. **Verify Python Interpreter:**

   - Ensure the virtual environment is selected as the Python interpreter (bottom right corner of VSCode).

7. **Run the Bluetooth Server Code:**

   - Execute `RPi/app.py`.
   - Check the output for the MAC Address of your Raspberry Pi's Bluetooth.

8. **Commit and Sync Changes:**

   - Commit your changes to the repository.  
      You'll have to set your git user and email
     > `git config --global user.name "FIRST_NAME LAST_NAME"`  
     > `git config --global user.email "MY_NAME@example.com"`
   - Sync the changes to the remote repository.

9. **Keep the Bluetooth Server Running:**
   - Ensure `RPi/app.py` continues to run while proceeding to the next steps.

See: https://youtu.be/sbLs92p8GKU

## Step 4: Clone the Classroom Repo to Your **Laptop** and Run AI Code + Connect to the Raspberry Pi via Bluetooth

We handle the BLE client connection via [Bleak](https://pypi.org/project/bleak/)

_"Bleak is a GATT client software, capable of connecting to BLE devices acting as GATT servers. It is designed to provide a asynchronous, cross-platform Python API to connect and communicate with e.g. sensors."_

1. **Ensure Python >=3.10 is Installed:**

   - Make sure you have Python >=3.10 installed on your laptop.

2. **Clone the Classroom Repository:**

   - Clone the same classroom repository to your laptop.  
     (again, using the HTTPS link)
   - Open the folder in VSCode.

3. **Set Up Python Virtual Environment:**

   - Open the Command Palette (`View > Command Palette`).
   - Search for `Python: Create Environment`.
   - Select `Venv` and choose a recent Python (e.g. 3.10) interpreter.
   - When prompted, check the `AI/requirements.txt` file and confirm by pressing OK.

If the venv doesn't work, manually install bleak via pip and run your code like in the AI labs.

4. **Modify AI Bluetooth Code:**

   - Open `AI/BLE_client.py` in VSCode.
   - Replace `"BLE_DEVICE_MAC"` on line 17 with the MAC address obtained from the Raspberry Pi.

5. **Run AI Bluetooth Code:**

   - Execute `AI/BLE.py` using the virtual environment.

   When running this file directly, it goes into demo mode and will send the current MMSS (minutes and seconds) via BLE

6. **Update AI Application Code:**

   - Open `AI/app.py`.
   - Update the `targetDeviceMac` on line 26 with the BLE MAC address of your Raspberry Pi.

7. **Run AI Application Code:**

   - Execute `AI/app.py`.
   - Check the console output.
   - Open the GradIO web interface.
   - Run a test video and verify if the values are printed in the console of the Raspberry Pi.  
     You can find a test video in _AI/testvideo/test1_default.mp4_

   The console will output a URL like "http://127.0.0.1:7860"  
   On this webpage you'll be able to upload images/videos to run analyze  
   The results will be sent via BLE to the RPi

8. **Commit and Sync Changes:**
   - Commit your changes to the repository.
   - Sync the changes to the remote repository.

## Step 5: Extend the Project with GPIO (LCD Display)

1. **Sync Git on the Raspberry Pi:**

   - Pull in the latest code from the repository on your Raspberry Pi (sync git).

Normally not needed as we won't change the AI code on the Pi and vice versa, but in case you've changed somethings in the docs folder e.g. this is the safe way.

2. **Connect and Use the LCD Display:**

   - Connect your LCD display to the Raspberry Pi.
   - Modify your code to print the messages received via Bluetooth onto the LCD display.  
     (and maybe also your BLE Mac address?)

3. **Run AI Application Code Again:**

   - Execute `AI/app.py` again.
   - Verify that the messages are appearing on the LCD display.

4. **Submit Your Work:**
   - Ensure you have [submitted the classroom repository in LeHo](https://leho-howest.instructure.com/courses/21067/assignments/187983).
   - Call an instructor to check out your working setup.

## Step 6: flag down a coach and show your working app's

- Show the working and completed kickoff-demo
  > Blue LED, Button and RGB Led
- Show AI code running on laptop communicating with RPi code, displaying values on LCD

---

## That's it for today! You've made a great start. Continue working on your project with these foundational steps. Good luck!

## ATTENTION: Own Project // venv // git

You will likely use your own venv for your own project, make sure to add this directory to your `.gitignore` file.
