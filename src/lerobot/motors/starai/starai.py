#!/usr/bin/env python

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

import logging
import time
from collections.abc import Sequence
from enum import Enum
from pprint import pformat
from typing import TYPE_CHECKING

from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode, MotorsBusBase, NameOrID, Value
from lerobot.utils.import_utils import _fashionstar_uart_sdk_available, require_package
from lerobot.utils.utils import enter_pressed, move_cursor_up

from .tables import MODEL_NUMBER_TABLE, MODEL_RESOLUTION

if TYPE_CHECKING or _fashionstar_uart_sdk_available:
    from fashionstar_uart_sdk.uart_pocket_handler import (
        PortHandler as StaraiPortHandler,
        SyncPositionControlOptions,
    )
else:
    StaraiPortHandler = None
    SyncPositionControlOptions = None

DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000
DEFAULT_ACC_TIME = 50
DEFAULT_DEC_TIME = 50
DEFAULT_MOTION_TIME = 350
DEFAULT_GRIPPER_MOTION_TIME = 100

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


class StaraiMotorsBus(MotorsBusBase):
    """Motor bus for Fashionstar StarAI arms over the UC-01 UART adapter."""

    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_number_table = MODEL_NUMBER_TABLE.copy()
    model_resolution_table = MODEL_RESOLUTION.copy()
    normalized_data = NORMALIZED_DATA.copy()

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        default_motion_time: int = DEFAULT_MOTION_TIME,
    ):
        require_package("fashionstar-uart-sdk", extra="starai", import_name="fashionstar_uart_sdk")
        super().__init__(port, motors, calibration)

        if StaraiPortHandler is None:
            raise ImportError("fashionstar-uart-sdk is required for StarAI motor support.")

        self.baudrate = baudrate
        self.default_motion_time = default_motion_time
        self.apply_drive_mode = True
        self.port_handler = StaraiPortHandler(port, baudrate)
        self._id_to_name_dict = {m.id: motor for motor, m in self.motors.items()}

    def __len__(self) -> int:
        return len(self.motors)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors:\n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")"
        )

    @property
    def is_connected(self) -> bool:
        return bool(self.port_handler.is_open)

    def connect(self, handshake: bool = True) -> None:
        try:
            self.port_handler.openPort()
            if handshake:
                self._handshake()
        except Exception as e:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from e

        self.disable_torque(mode="unlocked")
        self.port_handler.ResetLoop(0xFF)
        logger.debug("%s connected.", self.__class__.__name__)

    def _handshake(self) -> None:
        missing = []
        for motor, cfg in self.motors.items():
            if not self.port_handler.ping(cfg.id):
                missing.append(f"{motor} (id={cfg.id}, model={cfg.model})")
        if missing:
            raise RuntimeError(f"StarAI motor check failed on port '{self.port}': {missing}")

    def disconnect(self, disable_torque: bool = True) -> None:
        if not self.is_connected:
            return

        if disable_torque:
            self.disable_torque(num_retry=5)
            self.port_handler.clearPort()
        self.port_handler.closePort()
        logger.debug("%s disconnected.", self.__class__.__name__)

    def _id_to_name(self, motor_id: int) -> str:
        return self._id_to_name_dict[motor_id]

    def _get_motors_list(self, motors: NameOrID | Sequence[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self.motors)
        if isinstance(motors, str):
            return [motors]
        if isinstance(motors, int):
            return [self._id_to_name(motors)]
        if isinstance(motors, Sequence):
            return [m if isinstance(m, str) else self._id_to_name(m) for m in motors]
        raise TypeError(motors)

    def _get_values_dict(self, values: Value | dict[str, Value]) -> dict[str, Value]:
        if isinstance(values, (int, float)):
            return dict.fromkeys(self.motors, values)
        if isinstance(values, dict):
            return values
        raise TypeError(f"'values' is expected to be a single value or a dict. Got {values}")

    def read(self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0) -> Value:
        del num_retry
        if data_name == "Present_Position":
            return self.sync_read(data_name, motor, normalize=normalize)[motor]

        if data_name not in self.port_handler.read:
            raise NotImplementedError(f"StarAI does not support reading '{data_name}'.")
        return self.port_handler.read[data_name](self.motors[motor].id)

    def write(
        self,
        data_name: str,
        motor: str,
        value: Value,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> None:
        del num_retry
        if data_name == "Goal_Position":
            self.sync_write(data_name, {motor: value}, normalize=normalize)
            return

        raise NotImplementedError(f"StarAI does not support writing '{data_name}' through `write()`.")

    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
        num_retry: int = 0,
    ) -> dict[str, Value]:
        del num_retry
        if not self.is_connected:
            raise ConnectionError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"You need to run `{self.__class__.__name__}.connect()`."
            )

        names = self._get_motors_list(motors)
        if data_name not in ("Monitor", "Present_Position"):
            raise NotImplementedError(f"StarAI does not support sync-reading '{data_name}'.")

        servo_ids = {name: self.motors[name].id for name in names}
        monitor_data = self.port_handler.sync_read["Monitor"](servo_ids)
        raw_values = {}
        for name in names:
            position_deg = max(-180, min(180, monitor_data[name].current_position))
            raw_values[name] = int((position_deg + 180) / 360.0 * 4096)

        if normalize:
            return self._normalize(raw_values)
        return raw_values

    def sync_write(
        self,
        data_name: str,
        values: Value | dict[str, Value],
        *,
        normalize: bool = True,
        motion_time: int | None = None,
    ) -> None:
        if not self.is_connected:
            raise ConnectionError(
                f"{self.__class__.__name__}('{self.port}') is not connected. "
                f"You need to run `{self.__class__.__name__}.connect()`."
            )
        if data_name != "Goal_Position":
            raise NotImplementedError(f"StarAI does not support sync-writing '{data_name}'.")
        if SyncPositionControlOptions is None:
            raise ImportError("fashionstar-uart-sdk is required for StarAI motor support.")

        values_by_name = self._get_values_dict(values)
        raw_values = self._unnormalize(values_by_name) if normalize else values_by_name
        data_motion_time = self.default_motion_time if motion_time is None else motion_time

        write_data = {}
        for motor, raw_value in raw_values.items():
            target_position = int(((raw_value / 4096 * 360) - 180) * 10)
            write_data[motor] = SyncPositionControlOptions(
                self.motors[motor].id,
                target_position,
                data_motion_time,
                0,
                DEFAULT_ACC_TIME,
                DEFAULT_DEC_TIME,
            )

        if "gripper" in write_data:
            write_data["gripper"].power = 100 if self.motors["gripper"].model == "rx8-u50" else 1000
            write_data["gripper"].motion_time = DEFAULT_GRIPPER_MOTION_TIME

        self.port_handler.sync_write["Goal_Position"](write_data)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=DriveMode.NON_INVERTED.value,
                homing_offset=0,
                range_min=0,
                range_max=self.model_resolution_table.get(m.model, 4096) - 1,
            )
            for motor, m in self.motors.items()
        }

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        if cache:
            self.calibration = calibration_dict or {}

    def set_half_turn_homings(
        self, motors: NameOrID | Sequence[NameOrID] | None = None
    ) -> dict[NameOrID, Value]:
        motor_names = self._get_motors_list(motors)
        return dict.fromkeys(motor_names, 0)

    def record_ranges_of_motion(
        self, motors: NameOrID | Sequence[NameOrID] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        motor_names = self._get_motors_list(motors)
        start_positions = self.sync_read("Present_Position", motor_names, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", motor_names, normalize=False)
            mins = {motor: min(positions[motor], min_) for motor, min_ in mins.items()}
            maxes = {motor: max(positions[motor], max_) for motor, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for motor in motor_names:
                    print(f"{motor:<15} | {mins[motor]:>6} | {positions[motor]:>6} | {maxes[motor]:>6}")

            user_pressed_enter = enter_pressed()
            if display_values and not user_pressed_enter:
                move_cursor_up(len(motor_names) + 3)

        same_min_max = [motor for motor in motor_names if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def disable_torque(
        self,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        num_retry: int = 0,
        *,
        mode: str = "damping",
    ) -> None:
        del num_retry
        time.sleep(0.01)
        power = 900 if mode == "damping" else 0
        for motor in self._get_motors_list(motors):
            self.port_handler.write["Stop_On_Control_Mode"](self.motors[motor].id, mode, power)

    def enable_torque(self, motors: NameOrID | Sequence[NameOrID] | None = None, num_retry: int = 0) -> None:
        del num_retry
        for motor in self._get_motors_list(motors):
            self.port_handler.write["Stop_On_Control_Mode"](self.motors[motor].id, "locked", 0)

    def _normalize(self, values: dict[str, Value]) -> dict[str, Value]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for motor, val in values.items():
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[motor] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[motor] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table.get(self.motors[motor].model, 4096) - 1
                normalized_values[motor] = (bounded_val - mid) * 360 / max_res
            else:
                raise NotImplementedError

        return normalized_values

    def _unnormalize(self, values: dict[str, Value]) -> dict[str, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        raw_values = {}
        for motor, val in values.items():
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                raw_values[motor] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                raw_values[motor] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table.get(self.motors[motor].model, 4096) - 1
                raw_values[motor] = int((val * max_res / 360) + mid)
            else:
                raise NotImplementedError

        return raw_values
