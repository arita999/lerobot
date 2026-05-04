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

from lerobot.teleoperators.bi_starai_leader import BiStaraiLeader, BiStaraiLeaderConfig
from lerobot.teleoperators.starai_leader import StaraiViolin, StaraiViolinConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="StaraiLeaderBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    return bus


@pytest.fixture
def starai_leader_bus_patch():
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
        "lerobot.teleoperators.starai_leader.starai_leader.StaraiMotorsBus", side_effect=_bus_side_effect
    ):
        yield buses


def test_starai_violin_connect_and_get_action(starai_leader_bus_patch):
    teleop = StaraiViolin(StaraiViolinConfig(port="/dev/null"))

    teleop.connect(calibrate=False)
    assert teleop.is_connected

    action = teleop.get_action()
    assert set(action) == {f"{motor}.pos" for motor in teleop.bus.motors}

    teleop.disconnect()
    assert not teleop.is_connected


def test_make_teleoperator_from_starai_wiki_alias(starai_leader_bus_patch):
    teleop = make_teleoperator_from_config(StaraiViolinConfig(port="/dev/null"))

    assert isinstance(teleop, StaraiViolin)


def test_bi_starai_leader_prefixes_actions(starai_leader_bus_patch):
    teleop = BiStaraiLeader(
        BiStaraiLeaderConfig(left_arm_port="/dev/null-left", right_arm_port="/dev/null-right")
    )

    teleop.connect(calibrate=False)
    action = teleop.get_action()

    assert all(key.startswith(("left_", "right_")) for key in action)

    teleop.disconnect()
