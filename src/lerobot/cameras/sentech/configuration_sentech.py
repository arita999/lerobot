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

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["SentechCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("sentech")
@dataclass
class SentechCameraConfig(CameraConfig):
    """Configuration for Omron Sentech GigE Vision cameras via GenICam/GenTL.

    This backend uses the Python `harvesters` package and a GenTL CTI producer
    installed with the camera vendor SDK. For Omron Sentech cameras, point
    `cti_path` at the Sentech SDK CTI file, or set `LEROBOT_SENTECH_CTI`.
    """

    cti_path: str | Path | None = None
    serial_number_or_name: str | None = None
    device_index: int | None = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    timeout_ms: int = 1000
    pixel_format: str | None = None
    exposure_time_us: float | None = None
    gain: float | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.device_index is not None and self.serial_number_or_name is not None:
            raise ValueError("Specify either `device_index` or `serial_number_or_name`, not both.")

        if self.device_index is not None and self.device_index < 0:
            raise ValueError(f"`device_index` must be non-negative, but {self.device_index} is provided.")

        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, but {self.timeout_ms} is provided.")

        if (self.width is None) != (self.height is None):
            raise ValueError("`width` and `height` must either both be set, or both be omitted.")

        if self.exposure_time_us is not None and self.exposure_time_us <= 0:
            raise ValueError(
                f"`exposure_time_us` must be positive, but {self.exposure_time_us} is provided."
            )

        if self.gain is not None and self.gain < 0:
            raise ValueError(f"`gain` must be non-negative, but {self.gain} is provided.")
