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
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.starai import StaraiMotorsBus
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_starai_leader import StaraiViolinConfig

logger = logging.getLogger(__name__)

STARAI_LEADER_INITIAL_POSITION = {
    "Motor_0": 0,
    "Motor_1": -100,
    "Motor_2": 60,
    "Motor_3": 0,
    "Motor_4": 30,
    "Motor_5": 0,
    "gripper": 50,
}
STARAI_MOTOR_NAMES = ("Motor_0", "Motor_1", "Motor_2", "Motor_3", "Motor_4", "Motor_5")


class StaraiViolin(Teleoperator):
    """StarAI Violin leader arm."""

    config_class = StaraiViolinConfig
    name = "starai_violin"

    def __init__(self, config: StaraiViolinConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = StaraiMotorsBus(
            port=self.config.port,
            motors={
                **{motor: Motor(i, "rx8-u50", norm_mode_body) for i, motor in enumerate(STARAI_MOTOR_NAMES)},
                "gripper": Motor(6, "rx8-u50", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
            default_motion_time=1500,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        if self.is_calibrated:
            logger.info("%s slow start in progress, please wait for 1.5 seconds.", self)
            self.move_to_initial_position()

        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Writing calibration file associated with the id %s to the motors", self.id)
                self.bus.write_calibration(self.calibration)
                return

        logger.info("\nRunning calibration of %s", self)
        self.bus.disable_torque(mode="unlocked")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "For joints without limit stops, stay within 180 degrees clockwise/counterclockwise.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        logger.info("%s motors are configured by the StarAI firmware; skipping setup.", self)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s read action: %.1fms", self, dt_ms)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("StarAI Violin does not support force feedback yet.")

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def move_to_initial_position(self) -> RobotAction:
        self.bus.sync_write("Goal_Position", STARAI_LEADER_INITIAL_POSITION, motion_time=1500)
        time.sleep(1.5)
        self.bus.disable_torque(motors="gripper", mode="unlocked")
        return {f"{motor}.pos": val for motor, val in STARAI_LEADER_INITIAL_POSITION.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info("%s disconnected.", self)
