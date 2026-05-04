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

from pathlib import Path

import numpy as np
import pytest

from lerobot.cameras.sentech import SentechCamera, SentechCameraConfig
from lerobot.cameras.sentech import camera_sentech
from lerobot.utils.errors import DeviceNotConnectedError


class FakeNode:
    def __init__(self, value):
        self.value = value


class FakeNodeMap:
    def __init__(self):
        self.AcquisitionMode = FakeNode("Continuous")
        self.TriggerMode = FakeNode("Off")
        self.Width = FakeNode(2)
        self.Height = FakeNode(2)
        self.AcquisitionFrameRateEnable = FakeNode(False)
        self.AcquisitionFrameRate = FakeNode(30.0)
        self.PixelFormat = FakeNode("RGB8")
        self.ExposureTime = FakeNode(1000.0)
        self.Gain = FakeNode(0.0)


class FakeRemoteDevice:
    def __init__(self):
        self.node_map = FakeNodeMap()


class FakeDeviceInfo:
    id_ = "fake-id"
    serial_number = "SN123"
    display_name = "Fake Sentech"
    user_defined_name = "front"
    vendor = "OMRON SENTECH"
    model = "STC-Fake"
    tl_type = "GEV"
    version = "1.0"


class FakeComponent:
    width = 2
    height = 2
    data_format = "RGB8"
    data = np.arange(12, dtype=np.uint8)


class FakePayload:
    components = [FakeComponent()]


class FakeBuffer:
    payload = FakePayload()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeAcquirer:
    def __init__(self):
        self.remote_device = FakeRemoteDevice()
        self.started = False
        self.destroyed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def destroy(self):
        self.destroyed = True

    def fetch(self, timeout):
        return FakeBuffer()


class FakeHarvester:
    def __init__(self):
        self.files = []
        self.device_info_list = [FakeDeviceInfo()]
        self.acquirer = FakeAcquirer()
        self.reset_called = False

    def add_file(self, path):
        self.files.append(path)

    def update(self):
        pass

    def create(self, device_index):
        assert device_index == 0
        return self.acquirer

    def reset(self):
        self.reset_called = True


@pytest.fixture(autouse=True)
def patch_harvesters(monkeypatch, tmp_path):
    cti_path = tmp_path / "fake.cti"
    cti_path.write_text("fake")

    monkeypatch.setattr(camera_sentech, "Harvester", FakeHarvester)
    monkeypatch.setattr(camera_sentech, "GenTLTimeoutException", TimeoutError)
    monkeypatch.setattr(camera_sentech, "pfnc", None)
    monkeypatch.setattr(camera_sentech, "require_package", lambda *args, **kwargs: None)

    return cti_path


def test_config_validation():
    with pytest.raises(ValueError):
        SentechCameraConfig(cti_path=Path("fake.cti"), device_index=0, serial_number_or_name="SN123")

    with pytest.raises(ValueError):
        SentechCameraConfig(cti_path=Path("fake.cti"), width=640)


def test_find_cameras(patch_harvesters):
    cameras = SentechCamera.find_cameras(cti_path=patch_harvesters)

    assert cameras == [
        {
            "name": "OMRON SENTECH STC-Fake Fake Sentech",
            "type": "Sentech",
            "id": "SN123",
            "id_": "fake-id",
            "serial_number": "SN123",
            "display_name": "Fake Sentech",
            "user_defined_name": "front",
            "vendor": "OMRON SENTECH",
            "model": "STC-Fake",
            "tl_type": "GEV",
            "version": "1.0",
            "cti_path": str(patch_harvesters.resolve()),
        }
    ]


def test_connect_read_disconnect(patch_harvesters):
    config = SentechCameraConfig(
        cti_path=patch_harvesters,
        serial_number_or_name="SN123",
        width=2,
        height=2,
        fps=30,
        warmup_s=0,
    )

    with SentechCamera(config) as camera:
        assert camera.is_connected
        frame = camera.read()
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (2, 2, 3)
        assert frame.dtype == np.uint8

    assert not camera.is_connected


def test_read_before_connect(patch_harvesters):
    camera = SentechCamera(SentechCameraConfig(cti_path=patch_harvesters))

    with pytest.raises(DeviceNotConnectedError):
        camera.read()
