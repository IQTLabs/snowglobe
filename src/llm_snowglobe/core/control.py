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

import re
import random
import asyncio

from .intelligent import Intelligent
from .llm import LLM
from .player import Player
from .stateful import Stateful


class Control(Intelligent, Stateful):
    def __init__(
        self,
        database,
        verbosity,
        name,
        kind='ai',
        logger=None,
        source=None,
        model=None,
        menu=None,
        gen=None,
        embed=None,
        llm=None,
        reasoning=None,
        tools=None,
        ioid=None,
        iodict=None,
        presets=None,
        **kwargs
    ):
        # super().__init__(database=database, verbosity=verbosity, logger=logger, kind='ai', **kwargs)
        Intelligent.__init__(
            self,
            database=database,
            verbosity=verbosity, 
            kind=kind,
            name=name,
            iodict=iodict,
            logger=logger,
            **kwargs
        )
        Stateful.__init__(
            self,
            **kwargs
        )
        self.llm = (
            LLM(source=source, model=model, menu=menu, gen=gen, embed=embed)
            if llm is None
            else llm
        )
        self.name = "Control"
        self.persona = None
        self.reasoning = reasoning
        self.tools = tools
        self.ioid = ioid
        self.iodict = iodict
        self.presets = presets

    async def __call__(self):
        raise Exception(
            "! Override this method in the subclass for your specific scenario."
        )

    def run(self, *args, **kwargs):
        asyncio.run(self(*args, **kwargs))

    def header(self, title, h=0, width=80):
        print()
        if h == 0:
            print("+-" + "-" * min(len(title), width - 4) + "-+")
            print("| " + title + " |")
            print("+-" + "-" * min(len(title), width - 4) + "-+")
        elif h == 1:
            print("-" * min(len(title), width))
            print(title)
            print("-" * min(len(title), width))
        else:
            print(title)

    async def adjudicate(
        self,
        history=None,
        responses=None,
        query=None,
        nature=True,
        timestep="week",
        mode=[],
    ):
        responses_intro = "These are the plans for each person or group"
        if "geopol" in mode:
            responses_intro = "These are the plans ordered by each leader"
        if query is None:
            query = (
                "Weave these plans into a cohesive narrative of what happens in the next "
                + timestep
                + "."
            )
            if "geopol" in mode:
                query = "Describe these plans being carried out, assuming the leaders above issue no further orders."
            if random.random() < nature:
                query += " Include unexpected consequences."
        output = await self.return_output(
            history=history,
            responses=responses,
            responses_intro=responses_intro,
            query=query,
            query_format="oneline",
        )
        if "summarize" in mode:
            print("\n### Summary\n")
            template = "Give a short summary of the News.\n\n### History:\n\n{history}\n\n### News:\n\n{news}\n\n### Summary of the News:\n\n"
            variables = {"history": await history.textonly(), "news": output}
            output = self.return_output(template=template, variables=variables)
        return output

    async def assess(
        self, history=None, responses=None, query=None, mc=None, short=False
    ):
        responses_intro = "Questions about what happened"
        if responses is None:
            query_format = "twoline"
        else:
            query_format = "twoline_simple"
        bind = {"stop": ["\n\n"]} if short else None
        output = await self.return_output(
            bind=bind,
            history=history,
            history_over=True,
            responses=responses,
            responses_intro=responses_intro,
            query=query,
            query_format=query_format,
        )
        if mc is not None:
            output = await self.multiple_choice(query, output, mc)
        return output

    def chat(self, history=None):
        name = self.name
        persona = "the Control (a.k.a. moderator) of a simulated scenario"
        return self.chat_terminal(name=name, persona=persona, history=history)

    async def create_scenario(self, query=None, clip=0):
        if query is None:
            raise Exception("Query required to create scenario.")
        output = await self.return_output(query=query, query_format="twoline_simple")
        if clip > 0:
            output = "\n\n".join(output.split("\n\n")[:-clip])
        return output

    async def create_players(
        self,
        scenario,
        max_players=None,
        query=None,
        others=False,
        pattern_sep=None,
        pattern_left=None,
    ):
        if query is None:
            query = "List the key players in this scenario, separated by semicolons."
        if pattern_sep is None:
            pattern_sep = r"[\.\,;\n0-9]+"
        if pattern_left is None:
            pattern_left = " ()-"
        template = "Scenario: {scenario}\n\nQuestion: {query}\n\nAnswer: "
        variables = {"scenario": scenario, "query": query}
        output = await self.return_output(template=template, variables=variables)
        names = re.split(pattern_sep, output)
        names = [name.lstrip(pattern_left).rstrip() for name in names]
        names = [name for name in names if len(name) > 0]
        if max_players is None:
            player_names = names
            other_names = []
        else:
            player_names = names[:max_players]
            other_names = names[max_players:]
        players = [
            Player(llm=self.llm, name=name, persona=name) for name in player_names
        ]
        if not others:
            return players
        else:
            return players, other_names

    async def create_inject(self, history=None, query=None):
        if query is None:
            raise Exception("Query required to create inject.")
        output = await self.return_output(
            history=history,
            query=query,
            query_format="oneline",
            query_subtitle="Narrator",
        )
        return output
