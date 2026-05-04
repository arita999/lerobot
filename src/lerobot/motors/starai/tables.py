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

MODEL_NUMBER_TABLE = {
    "ra8-u25": 888,
    "ra8-u26": 889,
    "rx8-u25": 887,
    "rx8-u50": 999,
    "rx8-u51": 998,
    "rx18-u100": 1000,
}

MODEL_RESOLUTION = dict.fromkeys(MODEL_NUMBER_TABLE, 4096)
