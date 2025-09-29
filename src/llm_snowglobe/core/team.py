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
from .stateful import Stateful


class Team(Stateful):
    def __init__(self, name="Anonymous", leader=None, members=None, verbosity=2, logger=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.leader = leader
        self.members = members
        self.verbosity = verbosity

    async def respond(self, history=None, query=None, mc=None, short=False):
        member_responses = History()
        for member in self.members:
            if self.verbosity >= 2:
                print("\n### " + member.name)
            member_responses.add(
                member.name,
                await member.respond(history=history, query=query, mc=mc, short=short),
            )
        if self.verbosity >= 2:
            print("\n### Leader: " + self.leader.name)
        leader_response = await self.leader.synthesize(
            history=history, responses=member_responses, query=query, mc=mc
        )
        return leader_response

    async def synthesize(self, history=None, responses=None, query=None, mc=None):
        return await self.leader.synthesize(
            history=history, responses=responses, query=query, mc=mc
        )

    def chat(self, history=None):
        return self.leader.chat(history=history)

    def info(self, offset=0):
        print(" " * offset + "Team:", self.name)
        print(" " * offset + "  Leader:", self.leader.name)
        print(" " * offset + "  Members:", [member.name for member in self.members])
        if self.verbosity >= 2:
            for member in self.members:
                member.info(offset=offset + 2)
