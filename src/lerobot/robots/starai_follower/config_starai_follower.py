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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class StaraiFollowerConfig:
    """Base configuration for StarAI follower arms."""

    # Port to connect to the arm through the UC-01 adapter.
    port: str

    disable_torque_on_disconnect: bool = True

    # Safety cap for relative target changes. Set a scalar for every motor, or a per-motor dict.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # StarAI's current calibration flow uses the normalized [-100, 100] joint range by default.
    use_degrees: bool = False


@RobotConfig.register_subclass("starai_viola")
@RobotConfig.register_subclass("lerobot_robot_viola")
@dataclass
class StaraiViolaConfig(RobotConfig, StaraiFollowerConfig):
    pass


@RobotConfig.register_subclass("starai_cello")
@RobotConfig.register_subclass("lerobot_robot_cello")
@dataclass
class StaraiCelloConfig(RobotConfig, StaraiFollowerConfig):
    pass
