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

from .intelligent import Intelligent
from .stateful import Stateful

class Player(Intelligent, Stateful):
    def __init__(
        self,
        database,
        verbosity,
        logger=None,
        llm=None,
        name="Anonymous",
        kind="ai",
        persona=None,
        reasoning=None,
        tools=None,
        ioid=None,
        iodict=None,
        presets=None,
        **kwargs
    ):
        super().__init__(database=database, verbosity=verbosity, kind=kind, logger=logger, **kwargs)
        self.llm = llm
        self.name = name
        self.persona = persona
        self.reasoning = reasoning
        self.tools = tools
        self.ioid = ioid
        self.iodict = iodict
        self.presets = presets


    async def respond(self, history=None, query=None, reminder=2, mc=None, short=False):
        if query is None:
            query = "What action or actions do you take in response?"
        bind = {"stop": ["\n\n"]} if short else {"stop": ["Narrator:"]}
        output = await self.return_output(
            bind=bind,
            name=self.name,
            persona=self.persona,
            reminder=reminder,
            history=history,
            query=query,
        )
        if mc is not None:
            output = await self.multiple_choice(query, output, mc)
        return output

    async def synthesize(self, history=None, responses=None, query=None, mc=None):
        if query is None:
            responses_intro = (
                "These are the actions your team members recommend you take in response"
            )
            synthesize_query = "Combine the recommended actions given above"
        else:
            responses_intro = "These are the responses from your team members"
            synthesize_query = "Combine the responses given above"
        output = await self.return_output(
            name=self.name,
            persona=self.persona,
            history=history,
            responses=responses,
            responses_intro=responses_intro,
            query=synthesize_query,
            query_format="oneline",
        )
        if mc is not None:
            output = await self.multiple_choice(query, output, mc)
        return output

    def chat(self, history=None):
        name = self.name
        persona = self.persona
        return self.chat_terminal(name=name, persona=persona, history=history)

    def info(self, offset=0):
        print(" " * offset + "Player:", self.name)
        print(" " * offset + "  Type:", self.kind)
        print(" " * offset + "  Persona:", self.persona)
