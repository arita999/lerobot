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

# ruff: noqa: N802

from types import SimpleNamespace

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.starai import starai
from lerobot.motors.starai.starai import StaraiMotorsBus


class FakeSyncPositionControlOptions:
    def __init__(self, id: int, target_position: int, motion_time: int, power: int, t_acc: int, t_dec: int):
        self.id = id
        self.target_position = target_position
        self.motion_time = motion_time
        self.power = power
        self.t_acc = t_acc
        self.t_dec = t_dec


class FakePortHandler:
    instances = []

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self.is_open = False
        self.stop_commands = []
        self.goal_writes = []
        self.monitor_positions = {"Motor_0": 0.0, "gripper": 0.0}
        self.write = {"Stop_On_Control_Mode": self._stop_on_control_mode}
        self.read = {}
        self.sync_read = {"Monitor": self._sync_monitor}
        self.sync_write = {"Goal_Position": self._sync_goal_position}
        FakePortHandler.instances.append(self)

    def openPort(self):
        self.is_open = True

    def closePort(self):
        self.is_open = False

    def clearPort(self):
        pass

    def ping(self, id_: int) -> bool:
        return id_ in {0, 6}

    def ResetLoop(self, id_: int):
        self.reset_id = id_

    def _sync_monitor(self, motors: dict[str, int]):
        return {
            name: SimpleNamespace(id=id_, current_position=self.monitor_positions[name])
            for name, id_ in motors.items()
        }

    def _sync_goal_position(self, motors: dict[str, FakeSyncPositionControlOptions]):
        self.goal_writes.append(motors)

    def _stop_on_control_mode(self, id_: int, mode: str, power: int):
        self.stop_commands.append((id_, mode, power))


@pytest.fixture
def starai_bus(monkeypatch):
    FakePortHandler.instances.clear()
    monkeypatch.setattr(starai, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(starai, "StaraiPortHandler", FakePortHandler)
    monkeypatch.setattr(starai, "SyncPositionControlOptions", FakeSyncPositionControlOptions)

    motors = {
        "Motor_0": Motor(0, "ra8-u25", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "ra8-u25", MotorNormMode.RANGE_0_100),
    }
    calibration = {
        motor: MotorCalibration(id=m.id, drive_mode=0, homing_offset=0, range_min=0, range_max=4096)
        for motor, m in motors.items()
    }
    return StaraiMotorsBus("/dev/ttyUSB0", motors, calibration=calibration)


def test_connect_handshakes_and_unlocks(starai_bus):
    starai_bus.connect()

    assert starai_bus.is_connected
    handler = FakePortHandler.instances[-1]
    assert handler.reset_id == 0xFF
    assert handler.stop_commands == [(0, "unlocked", 0), (6, "unlocked", 0)]


def test_sync_read_normalizes_positions(starai_bus):
    starai_bus.connect()
    handler = FakePortHandler.instances[-1]
    handler.monitor_positions = {"Motor_0": 0.0, "gripper": 0.0}

    values = starai_bus.sync_read("Present_Position")

    assert values == {"Motor_0": 0.0, "gripper": 50.0}


def test_sync_write_unnormalizes_goal_positions(starai_bus):
    starai_bus.connect()

    starai_bus.sync_write("Goal_Position", {"Motor_0": 0.0, "gripper": 50.0})

    write_data = FakePortHandler.instances[-1].goal_writes[-1]
    assert write_data["Motor_0"].id == 0
    assert write_data["Motor_0"].target_position == 0
    assert write_data["gripper"].id == 6
    assert write_data["gripper"].target_position == 0
    assert write_data["gripper"].power == 1000
    assert write_data["gripper"].motion_time == 100
