# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Omron Sentech GigE Vision camera support via GenICam/GenTL.

The Python package used here, Harvesters, talks to a vendor GenTL producer
(`*.cti`). Omron Sentech provides such a producer through its camera SDK.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _harvesters_available, require_package

if TYPE_CHECKING or _harvesters_available:
    from genicam.gentl import TimeoutException as GenTLTimeoutException
    from harvesters.core import Harvester

    try:
        from harvesters.util import pfnc
    except Exception:  # pragma: no cover - defensive for older Harvesters versions
        pfnc = None
else:
    GenTLTimeoutException = TimeoutError
    Harvester = None
    pfnc = None

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_sentech import SentechCameraConfig

logger = logging.getLogger(__name__)

LEROBOT_SENTECH_CTI_ENV = "LEROBOT_SENTECH_CTI"
GENTL_CTI_ENV_VARS = ("GENICAM_GENTL64_PATH", "GENICAM_GENTL32_PATH", "GENICAM_GENTL_PATH")
COMMON_CTI_DIRS = (
    "/opt/sentech",
    "/opt/omron-sentech",
    "/opt/OMRON_SENTECH",
    "/usr/local/sentech",
    "/usr/local/omron-sentech",
    "C:/Program Files/Common Files/OMRON_SENTECH",
)


def _iter_path_entries(value: str | Path | None) -> Iterable[str | Path]:
    if value is None:
        return ()

    if isinstance(value, Path):
        return (value,)

    path = Path(value).expanduser()
    if path.exists():
        return (path,)

    return tuple(part for part in value.split(os.pathsep) if part)


def _expand_cti_paths(entries: Iterable[str | Path]) -> list[Path]:
    cti_paths: list[Path] = []

    for entry in entries:
        path = Path(entry).expanduser()
        if path.is_dir():
            cti_paths.extend(sorted(path.rglob("*.cti")))
        elif path.is_file() and path.suffix.lower() == ".cti":
            cti_paths.append(path)

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in cti_paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)

    return unique_paths


def resolve_cti_paths(cti_path: str | Path | None = None) -> list[Path]:
    """Resolve explicit, environment, and common SDK locations to CTI files."""

    entries: list[str | Path] = []
    entries.extend(_iter_path_entries(cti_path))
    entries.extend(_iter_path_entries(os.environ.get(LEROBOT_SENTECH_CTI_ENV)))

    for env_var in GENTL_CTI_ENV_VARS:
        entries.extend(_iter_path_entries(os.environ.get(env_var)))

    for directory in COMMON_CTI_DIRS:
        if Path(directory).exists():
            entries.append(directory)

    cti_paths = _expand_cti_paths(entries)
    if cti_paths:
        return cti_paths

    raise FileNotFoundError(
        "No GenTL CTI file was found for the Sentech camera. Set "
        f"`{LEROBOT_SENTECH_CTI_ENV}` or pass `cti_path`/`--cti-path` pointing to the CTI file "
        "installed with the Omron Sentech SDK."
    )


def _safe_device_info_value(device_info: Any, key: str) -> str | None:
    try:
        value = getattr(device_info, key)
    except Exception:
        return None

    if value is None:
        return None
    return str(value)


def _device_info_to_dict(device_info: Any, cti_path: Path | None = None) -> dict[str, Any]:
    keys = (
        "id_",
        "serial_number",
        "display_name",
        "user_defined_name",
        "vendor",
        "model",
        "tl_type",
        "version",
    )
    info = {key: value for key in keys if (value := _safe_device_info_value(device_info, key))}

    name_parts = [info.get("vendor"), info.get("model"), info.get("display_name")]
    name = " ".join(part for part in name_parts if part) or info.get("id_") or "Sentech GenICam Camera"
    camera_id = info.get("serial_number") or info.get("id_") or info.get("display_name")

    camera_info: dict[str, Any] = {
        "name": name,
        "type": "Sentech",
        "id": camera_id,
    }
    camera_info.update(info)
    if cti_path is not None:
        camera_info["cti_path"] = str(cti_path)
    return camera_info


class SentechCamera(Camera):
    """
    Captures frames from Omron Sentech GigE Vision cameras using a GenTL producer.

    Configure it with the `sentech` camera type:

    ```bash
    lerobot-record \
      --robot.cameras='{front: {type: sentech, cti_path: "/path/to/Sentech.cti", width: 640, height: 480, fps: 30}}'
    ```
    """

    def __init__(self, config: SentechCameraConfig):
        require_package("harvesters", extra="sentech")
        super().__init__(config)

        self.config = config
        self.cti_path = config.cti_path
        self.serial_number_or_name = config.serial_number_or_name
        self.device_index = config.device_index
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.timeout_ms = config.timeout_ms

        self.harvester: Harvester | None = None
        self.acquirer: Any | None = None
        self.device_info: dict[str, Any] | None = None
        self.cti_paths: list[Path] = []

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.capture_width: int | None = None
        self.capture_height: int | None = None

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        if self.device_info is not None:
            return f"{self.__class__.__name__}({self.device_info.get('id')})"
        if self.serial_number_or_name is not None:
            return f"{self.__class__.__name__}({self.serial_number_or_name})"
        if self.device_index is not None:
            return f"{self.__class__.__name__}(index={self.device_index})"
        return f"{self.__class__.__name__}(first)"

    @property
    def is_connected(self) -> bool:
        return self.acquirer is not None

    @staticmethod
    def _create_harvester(cti_path: str | Path | None = None) -> tuple[Harvester, list[Path]]:
        if Harvester is None:
            raise ImportError("The `harvesters` package is required for Sentech cameras.")

        cti_paths = resolve_cti_paths(cti_path)
        harvester = Harvester()
        for path in cti_paths:
            harvester.add_file(str(path))
        harvester.update()
        return harvester, cti_paths

    @staticmethod
    def find_cameras(cti_path: str | Path | None = None) -> list[dict[str, Any]]:
        harvester, cti_paths = SentechCamera._create_harvester(cti_path)
        try:
            found_cameras_info = []
            cti_for_metadata = cti_paths[0] if cti_paths else None
            for device_info in harvester.device_info_list:
                found_cameras_info.append(_device_info_to_dict(device_info, cti_for_metadata))
            return found_cameras_info
        finally:
            harvester.reset()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self.harvester, self.cti_paths = self._create_harvester(self.cti_path)

        device_infos = list(self.harvester.device_info_list)
        if not device_infos:
            self._cleanup()
            raise ConnectionError(
                "No Sentech/GenICam camera was detected. Check camera power, network configuration, "
                "and the CTI path."
            )

        device_index = self._select_device_index(device_infos)
        self.device_info = _device_info_to_dict(
            device_infos[device_index], self.cti_paths[0] if self.cti_paths else None
        )
        self.acquirer = self._create_image_acquirer(device_index)

        try:
            self._configure_capture_settings()
            self._start_acquisition()
            self._start_read_thread()

            if warmup and self.warmup_s > 0:
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                    time.sleep(0.1)
                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")
        except Exception:
            self._cleanup()
            raise

        logger.info(f"{self} connected.")

    def _select_device_index(self, device_infos: Sequence[Any]) -> int:
        if self.device_index is not None:
            if self.device_index >= len(device_infos):
                raise ValueError(
                    f"Sentech device_index={self.device_index} is out of range. "
                    f"Detected {len(device_infos)} camera(s)."
                )
            return self.device_index

        if self.serial_number_or_name is None:
            return 0

        matches = []
        expected = self.serial_number_or_name
        for idx, device_info in enumerate(device_infos):
            info = _device_info_to_dict(device_info)
            values = {str(value) for value in info.values() if value is not None}
            if expected in values:
                matches.append(idx)

        if not matches:
            available = [_device_info_to_dict(info) for info in device_infos]
            raise ValueError(
                f"No Sentech camera matched serial/name '{expected}'. Available devices: {available}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple Sentech cameras matched serial/name '{expected}'. Use `device_index` instead."
            )

        return matches[0]

    def _create_image_acquirer(self, device_index: int) -> Any:
        if self.harvester is None:
            raise DeviceNotConnectedError(f"{self} harvester is not initialized")

        if hasattr(self.harvester, "create"):
            return self.harvester.create(device_index)

        return self.harvester.create_image_acquirer(list_index=device_index)

    def _node_map(self) -> Any | None:
        if self.acquirer is None:
            return None
        remote_device = getattr(self.acquirer, "remote_device", None)
        return getattr(remote_device, "node_map", None)

    @staticmethod
    def _read_node_value(node_map: Any, names: Sequence[str]) -> Any | None:
        for name in names:
            node = getattr(node_map, name, None)
            if node is None:
                continue
            try:
                return getattr(node, "value", None)
            except Exception:
                continue
        return None

    @staticmethod
    def _set_node_value(node_map: Any, names: Sequence[str], value: Any, *, required: bool) -> bool:
        errors = []
        for name in names:
            node = getattr(node_map, name, None)
            if node is None:
                continue
            try:
                node.value = value
                return True
            except Exception as e:
                errors.append(f"{name}: {e}")

        message = f"Could not set GenICam node(s) {tuple(names)} to {value!r}"
        if errors:
            message = f"{message}: {'; '.join(errors)}"

        if required:
            raise RuntimeError(message)

        logger.debug(message)
        return False

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        node_map = self._node_map()
        if node_map is None:
            logger.warning(f"{self} has no GenICam node map; using camera defaults.")
            return

        self._set_node_value(node_map, ("AcquisitionMode",), "Continuous", required=False)
        self._set_node_value(node_map, ("TriggerMode",), "Off", required=False)

        if self.config.pixel_format is not None:
            self._set_node_value(node_map, ("PixelFormat",), self.config.pixel_format, required=True)

        if self.capture_width is not None and self.capture_height is not None:
            self._set_node_value(node_map, ("Width",), int(self.capture_width), required=True)
            self._set_node_value(node_map, ("Height",), int(self.capture_height), required=True)
        else:
            actual_width = self._read_node_value(node_map, ("Width",))
            actual_height = self._read_node_value(node_map, ("Height",))
            if actual_width is not None and actual_height is not None:
                self.capture_width = int(actual_width)
                self.capture_height = int(actual_height)
                if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    self.width, self.height = self.capture_height, self.capture_width
                else:
                    self.width, self.height = self.capture_width, self.capture_height

        if self.fps is not None:
            self._set_node_value(node_map, ("AcquisitionFrameRateEnable",), True, required=False)
            fps_set = self._set_node_value(
                node_map,
                ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
                float(self.fps),
                required=False,
            )
            if not fps_set:
                logger.warning(f"{self} could not set fps={self.fps}; continuing with camera default.")
        else:
            actual_fps = self._read_node_value(node_map, ("AcquisitionFrameRate", "ResultingFrameRate"))
            if actual_fps is not None:
                self.fps = int(round(float(actual_fps)))

        if self.config.exposure_time_us is not None:
            self._set_node_value(
                node_map,
                ("ExposureTime", "ExposureTimeAbs"),
                float(self.config.exposure_time_us),
                required=True,
            )

        if self.config.gain is not None:
            self._set_node_value(node_map, ("Gain", "GainRaw"), float(self.config.gain), required=True)

    def _start_acquisition(self) -> None:
        if self.acquirer is None:
            raise DeviceNotConnectedError(f"{self} acquirer is not initialized")

        if hasattr(self.acquirer, "start"):
            self.acquirer.start()
        else:
            self.acquirer.start_acquisition()

    def _stop_acquisition(self) -> None:
        if self.acquirer is None:
            return

        if hasattr(self.acquirer, "stop"):
            self.acquirer.stop()
        else:
            self.acquirer.stop_acquisition()

    @staticmethod
    def _data_format_name(data_format: Any) -> str:
        if pfnc is not None:
            format_by_int = getattr(pfnc, "dict_by_ints", {})
            try:
                pfnc_format = format_by_int.get(int(data_format))
                name = getattr(pfnc_format, "name", None)
                if name:
                    return str(name)
                if pfnc_format is not None:
                    return str(pfnc_format)
            except Exception:
                pass

        name = getattr(data_format, "name", None)
        if name:
            return str(name)
        return str(data_format)

    @staticmethod
    def _format_in(data_format: Any, data_format_name: str, collection_name: str) -> bool:
        collection = getattr(pfnc, collection_name, ()) if pfnc is not None else ()
        if data_format in collection:
            return True
        return any(data_format_name == getattr(item, "name", str(item)) for item in collection)

    @staticmethod
    def _bit_depth(data_format_name: str, default: int = 8) -> int:
        match = re.search(r"(\d+)(?:Packed)?$", data_format_name)
        if match is None:
            return default
        return int(match.group(1))

    @classmethod
    def _to_uint8(cls, image: NDArray[Any], data_format_name: str) -> NDArray[Any]:
        if image.dtype == np.uint8:
            return image

        bit_depth = cls._bit_depth(data_format_name, default=int(image.dtype.itemsize * 8))
        max_value = float((1 << bit_depth) - 1)
        return np.clip(image.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)

    @staticmethod
    def _bayer_conversion_code(data_format_name: str) -> int | None:
        if "BayerRG" in data_format_name:
            return int(cv2.COLOR_BayerRG2RGB)
        if "BayerGR" in data_format_name:
            return int(cv2.COLOR_BayerGR2RGB)
        if "BayerGB" in data_format_name:
            return int(cv2.COLOR_BayerGB2RGB)
        if "BayerBG" in data_format_name:
            return int(cv2.COLOR_BayerBG2RGB)
        return None

    @classmethod
    def _component_to_rgb(cls, component: Any) -> NDArray[Any]:
        width = int(component.width)
        height = int(component.height)
        data_format = component.data_format
        data_format_name = cls._data_format_name(data_format)
        data = np.asarray(component.data)

        is_mono = cls._format_in(
            data_format, data_format_name, "mono_location_formats"
        ) or data_format_name.startswith("Mono")
        is_rgb = cls._format_in(data_format, data_format_name, "rgb_formats") or data_format_name.startswith(
            "RGB"
        )
        is_bgr = cls._format_in(data_format, data_format_name, "bgr_formats") or data_format_name.startswith(
            "BGR"
        )
        is_rgba = cls._format_in(
            data_format, data_format_name, "rgba_formats"
        ) or data_format_name.startswith("RGBA")
        is_bgra = cls._format_in(
            data_format, data_format_name, "bgra_formats"
        ) or data_format_name.startswith("BGRA")

        bayer_code = cls._bayer_conversion_code(data_format_name)
        if bayer_code is not None:
            raw = cls._to_uint8(data.reshape(height, width), data_format_name)
            return cv2.cvtColor(raw, bayer_code)

        if is_mono:
            gray = cls._to_uint8(data.reshape(height, width), data_format_name)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if is_rgb or is_bgr:
            image = cls._to_uint8(data.reshape(height, width, 3), data_format_name)
            if is_bgr:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        if is_rgba or is_bgra:
            image = cls._to_uint8(data.reshape(height, width, 4), data_format_name)
            if is_bgra:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return image

        if data.size == height * width * 3:
            logger.warning(f"Unknown pixel format {data_format_name}; assuming RGB8 layout.")
            return cls._to_uint8(data.reshape(height, width, 3), data_format_name)

        raise RuntimeError(
            f"Unsupported Sentech pixel format {data_format_name}. Configure the camera with "
            "`pixel_format: RGB8`, `BGR8`, `Mono8`, or a Bayer8 format if available."
        )

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.acquirer is None:
            raise DeviceNotConnectedError(f"{self} acquirer is not initialized")

        try:
            with self.acquirer.fetch(timeout=self.timeout_ms / 1000.0) as buffer:
                component = buffer.payload.components[0]
                return self._component_to_rgb(component).copy()
        except GenTLTimeoutException as e:
            raise TimeoutError(f"{self} timeout after {self.timeout_ms}ms") from e

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"{self} frame shape={image.shape} does not match expected HxWx3.")

        h, w = image.shape[:2]
        if self.capture_width is None or self.capture_height is None:
            self.capture_width, self.capture_height = w, h
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = h, w
            else:
                self.width, self.height = w, h

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        return processed_image

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware()
                processed_frame = self._postprocess_image(raw_frame)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def _cleanup(self) -> None:
        if self.thread is not None:
            self._stop_read_thread()

        if self.acquirer is not None:
            try:
                self._stop_acquisition()
            except Exception:
                logger.debug(f"{self} failed to stop acquisition cleanly.", exc_info=True)

            try:
                if hasattr(self.acquirer, "destroy"):
                    self.acquirer.destroy()
            finally:
                self.acquirer = None

        if self.harvester is not None:
            self.harvester.reset()
            self.harvester = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self._cleanup()
        logger.info(f"{self} disconnected.")
