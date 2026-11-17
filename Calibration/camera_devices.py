"""Windows DirectShow camera discovery."""

from dataclasses import dataclass
import json
import subprocess


@dataclass(frozen=True)
class Camera:
    index: int
    name: str
    device_path: str = ""
    instance_id: str = ""
    hardware_ids: tuple[str, ...] = ()

    @property
    def label(self):
        identity = ", ".join(self.hardware_ids) or "hardware ID unavailable"
        return f"{self.name} (index {self.index}; {identity})"


def instance_from_path(path):
    """Convert a standard DirectShow device interface path to a PnP instance ID."""
    parts = path.split("#")
    if len(parts) < 4 or not parts[0].startswith("\\\\?\\"):
        return ""
    return "\\".join((parts[0][4:], parts[1], parts[2]))


def attach_hardware_ids(devices, entries):
    hardware = {}
    for entry in entries:
        identifiers = entry["HardwareID"]
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        hardware[entry["PNPDeviceID"].casefold()] = tuple(identifiers)
    return [Camera(c.index, c.name, c.device_path, c.instance_id,
                   hardware.get(c.instance_id.casefold(), ())) for c in devices]


def discover_cameras():
    # DirectShow enumeration order is used by OpenCV's CAP_DSHOW backend, but
    # device changes between enumeration and opening can still change indices.
    import comtypes
    from comtypes import GUID, COMError
    from comtypes.persist import IPropertyBag
    from pygrabber.dshow_graph import SystemDeviceEnum
    from pygrabber.dshow_ids import DeviceCategories

    comtypes.CoInitialize()
    try:
        enumerator = SystemDeviceEnum().system_device_enum.CreateClassEnumerator(
            GUID(DeviceCategories.VideoInputDevice), dwFlags=0
        )
        devices = []
        if enumerator:
            while True:
                try:
                    moniker, count = enumerator.Next(1)
                except ValueError:
                    break
                if not count:
                    break
                bag = moniker.BindToStorage(0, 0, IPropertyBag._iid_).QueryInterface(IPropertyBag)
                name = str(bag.Read("FriendlyName", pErrorLog=None))
                try:
                    path = str(bag.Read("DevicePath", pErrorLog=None))
                except (COMError, OSError):
                    path = ""
                devices.append(Camera(len(devices), name, path, instance_from_path(path)))
    finally:
        comtypes.CoUninitialize()

    # Match only an exact PnP instance ID, never a friendly name or model ID.
    if not any(camera.instance_id for camera in devices):
        return devices, "Hardware identifiers are unavailable for these devices."
    script = (
        "$ErrorActionPreference='Stop'; "
        "Get-CimInstance -ClassName Win32_PnPEntity | "
        "Where-Object { $_.PNPDeviceID -and $_.HardwareID } | "
        "Select-Object PNPDeviceID, HardwareID | ConvertTo-Json -Compress -Depth 4"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        data = json.loads(result.stdout) if result.stdout.strip() else []
        if isinstance(data, dict):
            data = [data]
        devices = attach_hardware_ids(devices, data)
        return devices, ""
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as exc:
        return devices, f"Could not retrieve Windows hardware IDs: {exc}"
