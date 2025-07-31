#!/usr/bin/env python3
"""
bluetooth_uart_server.py  –  Thread-friendly Nordic-UART GATT server.

Public function
~~~~~~~~~~~~~~~
    ble_gatt_uart_loop(rx_q, tx_q, device_name, events_q=None)

* runs a GLib main-loop forever (meant for a background thread)
* enqueues received packets onto `rx_q`
* sends strings taken from `tx_q`
* if `events_q` is supplied, emits tuples:

      ("connected", True/False)

  whenever BlueZ’s Device1.Connected property changes.
"""

from __future__ import annotations
import os, queue, threading

# ---------- BlueZ / GLib imports -------------------------------------------
import dbus, dbus.mainloop.glib
from gi.repository import GLib

# Nordic-UART UUIDs
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_UUID      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_UUID      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

BLUEZ = "org.bluez"
OM_IFACE  = "org.freedesktop.DBus.ObjectManager"
LE_ADV_IF = "org.bluez.LEAdvertisingManager1"
GATT_MGR  = "org.bluez.GattManager1"
CHRC_IF   = "org.bluez.GattCharacteristic1"

# ---------- Minimal BlueZ helper classes (unchanged) ------------------------
from .utils_advertisement import Advertisement, register_ad_cb, register_ad_error_cb
from .utils_gatt_server   import Characteristic, Service, register_app_cb, register_app_error_cb



def get_ble_mac():
    """Return primary adapter’s Bluetooth MAC, or 'unknown'."""
    import dbus
    import dbus.mainloop.glib                       # ensure bus has a loop
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()
    om  = dbus.Interface(bus.get_object("org.bluez", "/"),
                         "org.freedesktop.DBus.ObjectManager")
    for path, ifaces in om.GetManagedObjects().items():
        a = ifaces.get("org.bluez.Adapter1")
        if a and "Address" in a:
            return str(a["Address"])
    return "unknown"



class TxCharacteristic(Characteristic):
    def __init__(self, bus, service, tx_q):
        super().__init__(bus, 0, UART_TX_UUID, ["notify"], service)
        self._q, self._notify = tx_q, False
    def StartNotify(self):                                     # noqa: N802
        if self._notify: return
        self._notify = True
        GLib.timeout_add(20, self._flush)
    def StopNotify(self): self._notify = False                 # noqa: N802
    def _flush(self):
        if not self._notify: return False
        try:
            while True:
                payload = self._q.get_nowait()
                self.PropertiesChanged(
                    CHRC_IF,
                    {"Value": [dbus.Byte(b) for b in payload.encode()]},
                    [],
                )
        except queue.Empty:
            pass
        return True

class RxCharacteristic(Characteristic):
    def __init__(self, bus, service, rx_q):
        super().__init__(bus, 1, UART_RX_UUID, ["write"], service)
        self._q = rx_q
    def WriteValue(self, value, _options):                      # noqa: N802
        self._q.put(bytearray(value).decode())

class UartService(Service):
    def __init__(self, bus, rx_q, tx_q):
        super().__init__(bus, 0, UART_SERVICE_UUID, True)
        self.add_characteristic(TxCharacteristic(bus, self, tx_q))
        self.add_characteristic(RxCharacteristic(bus, self, rx_q))

class UartApp(dbus.service.Object):
    PATH = "/"
    def __init__(self, bus, srv):
        super().__init__(bus, self.PATH)
        self._objs = {srv.get_path(): srv.get_properties()}
        for ch in srv.get_characteristics():
            self._objs[ch.get_path()] = ch.get_properties()
    @dbus.service.method(OM_IFACE, out_signature="a{oa{sa{sv}}}")  # noqa: N802
    def GetManagedObjects(self):
        return self._objs

    def get_path(self):
        return dbus.ObjectPath(self.PATH)

class UartAdv(Advertisement):
    def __init__(self, bus, name):
        super().__init__(bus, 0, "peripheral")
        self.add_service_uuid(UART_SERVICE_UUID)
        self.add_local_name(name)
        self.include_tx_power = True

# ---------- Adapter helper --------------------------------------------------
def _find_adapter(bus):
    om = dbus.Interface(bus.get_object(BLUEZ, "/"), OM_IFACE)
    for p, props in om.GetManagedObjects().items():
        if LE_ADV_IF in props and GATT_MGR in props:
            return p
    return None

# ---------- Public entry-point ---------------------------------------------
def ble_gatt_uart_loop(rx_q: "queue.Queue[str]",
                       tx_q: "queue.Queue[str]",
                       device_name: str,
                       events_q: "queue.Queue" | None = None) -> None:
    """
    Blocks forever: starts Nordic-UART GATT server & advert.
    """
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus  = dbus.SystemBus()

    adapter = _find_adapter(bus)
    if not adapter:
        raise RuntimeError("No BLE adapter with GATT & advertising managers")

    # power on if needed
    props = dbus.Interface(bus.get_object(BLUEZ, adapter), "org.freedesktop.DBus.Properties")
    if not props.Get("org.bluez.Adapter1", "Powered"):
        props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(1))

    srv = UartService(bus, rx_q, tx_q)
    app = UartApp(bus, srv)
    adv = UartAdv(bus, device_name)

    srv_mgr = dbus.Interface(bus.get_object(BLUEZ, adapter), GATT_MGR)
    adv_mgr = dbus.Interface(bus.get_object(BLUEZ, adapter), LE_ADV_IF)

    srv_mgr.RegisterApplication(app.get_path(), {},
                                reply_handler=lambda: print("[ble] GATT registered"),
                                error_handler=lambda e: print("[ble] GATT err:", e))

    adv_mgr.RegisterAdvertisement(adv.get_path(), {},
                                  reply_handler=lambda: print("[ble] Advert registered"),
                                  error_handler=lambda e: print("[ble] Advert err:", e))

        # --- Connection-state signal ----------------------------------------
    if events_q is not None:
        def _prop_changed(iface, changed, _invalid, path):
            if iface == "org.bluez.Device1" and "Connected" in changed:
                # print("putting in evt q", bool(changed["Connected"]))
                # print("changed", changed)
                events_q.put(bool(changed["Connected"]))

        bus.add_signal_receiver(
            _prop_changed,
            dbus_interface="org.freedesktop.DBus.Properties",
            signal_name="PropertiesChanged",
            arg0="org.bluez.Device1",
            path_keyword="path",
        )

        # ───────────────► emit current state once ◄───────────────
        dev_iface = "org.bluez.Device1"
        om = dbus.Interface(bus.get_object(BLUEZ, "/"), OM_IFACE)
        for p, props in om.GetManagedObjects().items():
            if dev_iface in props and props[dev_iface].get("Connected"):
                events_q.put(True)
                break


    print(f"[ble] Advertising as “{device_name}” (thread {threading.current_thread().name})")
    GLib.MainLoop().run()
