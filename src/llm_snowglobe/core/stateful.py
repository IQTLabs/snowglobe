#!/usr/bin/env python3

#   Copyright 2023-2025 IQT Labs LLC
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from .history import History


class Stateful:
    def __init__(self, **kwargs):
        self.history = History()

    def record_narration(self, narration, timestep=None, index=None):
        if timestep is None and index is None:
            label = "Narrator"
        else:
            part1 = timestep.title() + " " if timestep is not None else ""
            part2 = str(index) if index is not None else str(len(self.history))
            label = part1 + part2
        self.history.add(label, narration)

    def record_response(self, player_name, player_response):
        self.history.add(player_name, player_response)
