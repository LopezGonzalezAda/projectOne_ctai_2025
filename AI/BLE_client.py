import asyncio
from itertools import count, takewhile
import sys
import time
from queue import Queue
import threading
from datetime import datetime
from typing import Iterator

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

# BLE UART UUIDs (Nordic spec)
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # write
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # notify

BLE_DEVICE_MAC = "2C:CF:67:9B:2B:A0"

# Shared queues
tx_q = Queue()
rx_q = Queue()

def sliced(data: bytes, n: int) -> Iterator[bytes]:
    return takewhile(len, (data[i: i + n] for i in count(0, n)))

async def uart_terminal(rx_q=None, tx_q=None, targetDeviceMac=None):
    def match_nus_uuid(device: BLEDevice, adv: AdvertisementData):
        print("found", device.address)
        return device.address == targetDeviceMac and UART_SERVICE_UUID.lower() in adv.service_uuids

    device = await BleakScanner.find_device_by_filter(match_nus_uuid)

    if device is None:
        print("❌ No matching BLE device found.")
        sys.exit(1)

    print("✅ Found device, connecting...")

    def handle_disconnect(_: BleakClient):
        print("❌ BLE device disconnected.")
        for task in asyncio.all_tasks():
            task.cancel()

    def handle_rx(_: BleakGATTCharacteristic, data: bytearray):
        print("📩 Received:", data)
        if rx_q:
            rx_q.put(data)

    async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
        try:
            print("🔗 Connected")
            await asyncio.sleep(2)  # allow service discovery to settle

            await client.start_notify(UART_TX_CHAR_UUID, handle_rx)

            while True:
                try:
                    data = tx_q.get_nowait()
                    if data:
                        encoded = data.encode()
                        for chunk in sliced(encoded, 20):
                            await client.write_gatt_char(UART_RX_CHAR_UUID, chunk, response=False)
                        print("📤 Sent:", data)
                except:
                    await asyncio.sleep(0.1)

        except Exception as e:
            print("❌ BLE loop error:", e)

def run(rx_q=None, tx_q=None, targetDeviceMac=None):
    if targetDeviceMac is None:
        raise ValueError("targetDeviceMac cannot be None")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(uart_terminal(rx_q=rx_q, tx_q=tx_q, targetDeviceMac=targetDeviceMac))
    except asyncio.exceptions.CancelledError:
        pass

def init_ble_thread():
    thread = threading.Thread(target=run, args=(rx_q, tx_q, BLE_DEVICE_MAC), daemon=True)
    thread.start()
