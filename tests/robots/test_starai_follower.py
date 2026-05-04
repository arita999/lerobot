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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_starai_follower import BiStaraiFollower, BiStaraiFollowerConfig
from lerobot.robots.starai_follower import StaraiCello, StaraiCelloConfig, StaraiViola, StaraiViolaConfig
from lerobot.robots.utils import make_robot_from_config


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="StaraiBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    return bus


@pytest.fixture
def starai_bus_patch():
    buses = []

    def _bus_side_effect(*_args, **kwargs):
        bus = _make_bus_mock()
        bus.motors = kwargs["motors"]
        motors_order = list(bus.motors)
        bus.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus.sync_write.return_value = None
        buses.append(bus)
        return bus

    with patch(
        "lerobot.robots.starai_follower.starai_follower.StaraiMotorsBus", side_effect=_bus_side_effect
    ):
        yield buses


def test_starai_viola_connect_observe_and_act(starai_bus_patch):
    robot = StaraiViola(StaraiViolaConfig(port="/dev/null"))

    robot.connect(calibrate=False)
    assert robot.is_connected

    obs = robot.get_observation()
    assert set(obs) == {f"{motor}.pos" for motor in robot.bus.motors}

    action = {f"{motor}.pos": i * 10 for i, motor in enumerate(robot.bus.motors, 1)}
    returned = robot.send_action(action)
    assert returned == action
    robot.bus.sync_write.assert_called_with(
        "Goal_Position", {motor: i * 10 for i, motor in enumerate(robot.bus.motors, 1)}
    )

    robot.disconnect()
    assert not robot.is_connected


def test_starai_cello_uses_cello_model(starai_bus_patch):
    robot = StaraiCello(StaraiCelloConfig(port="/dev/null"))

    assert {motor.model for motor in robot.bus.motors.values()} == {"rx8-u25"}


def test_make_robot_from_starai_wiki_alias(starai_bus_patch):
    robot = make_robot_from_config(StaraiViolaConfig(port="/dev/null"))

    assert isinstance(robot, StaraiViola)


def test_bi_starai_follower_prefixes_actions(starai_bus_patch):
    robot = BiStaraiFollower(
        BiStaraiFollowerConfig(
            left_arm_port="/dev/null-left",
            right_arm_port="/dev/null-right",
            arm_name="starai_viola",
        )
    )

    robot.connect(calibrate=False)
    obs = robot.get_observation()

    assert all(key.startswith(("left_", "right_")) for key in obs)
    action = {feature: i * 10 for i, feature in enumerate(robot.action_features, 1)}
    returned = robot.send_action(action)

    assert returned == action
    assert starai_bus_patch[0].sync_write.call_args.args[1] == {
        key.removeprefix("left_").removesuffix(".pos"): value
        for key, value in action.items()
        if key.startswith("left_")
    }
    assert starai_bus_patch[1].sync_write.call_args.args[1] == {
        key.removeprefix("right_").removesuffix(".pos"): value
        for key, value in action.items()
        if key.startswith("right_")
    }

    robot.disconnect()
