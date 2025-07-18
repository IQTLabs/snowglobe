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

import llm_snowglobe as snowglobe


class AzuristanCrimsonia(snowglobe.Control):
    def __init__(self):
        super().__init__()

        goals = {
            'azuristan_dove': "Your goal is to avoid war at all costs, and to preserve the sovereignty of Azuristan if possible.",
            'azuristan_hawk': "Your goal is to preserve the sovereignty of Azuristan, even if it requires starting a war.",
            'crimsonia_dove': "Your goal is to avoid war at all costs, and to unify the Crimsonian people if possible.",
            'crimsonia_hawk': "Your goal is to unify the Crimsonian people, even if it requires starting a war.",
        }

        self.title = 'Azuristan and Crimsonia'
        self.players = [
            snowglobe.Player(
                llm=self.llm,
                name='President of Azuristan',
                persona='the leader of Azuristan.  {}'.format(goals[
                    'azuristan_dove'])),
            snowglobe.Player(
                llm=self.llm,
                name='Premier of Crimsonia',
                persona='the leader of Crimsonia.  {}'.format(goals[
                    'crimsonia_dove'])),
        ]
        self.scenario = """\
Azuristan and Crimsonia are neighboring countries in Central Asia.  Azuristan is a Western-backed democracy that suffers from endemic corruption.  Crimsonia is controlled by an autocratic government that stifles dissent and commits human rights violations.

Both countries have modern professional militaries, although Crimsonia's is slightly larger than Azuristan's.  In addition, Azuristan possesses ten nuclear weapons and Crimsonia has eight.

Most citizens of Azuristan are from the Azuristani ethnic group, and most citizens of Crimsonia are from the Crimsonian ethnic group.  The only exception is Azuristan's province of Tyriana, on the border with Crimsonia.  Most residents of Tyriana belong to the Crimsonian ethnic group.

The animosity between Azuristan and Crimsonia extends back over centuries of ethnic tension and intermittent warfare.  Recent years have been fairly calm.  However, that suddenly changes when Tyriana declares independence.  Amidst the crisis, local leaders in Tyriana ask Crimsonia to come to the province's defense, and the same leaders indicate that they want Tyriana to become part of Crimsonia."""
        self.moves = 3
        self.timestep = 'month'
        self.nature = True
        self.mode = ['geopol']
        self.questions = [
            "In one sentence, what was the outcome?",
        ]
        self.mc_questions = [
            ["What was the final status of Tyriana?", ['part of Azuristan', 'part of Crimsonia', 'independent', 'not yet determined']],
            ["Did armed conflict occur?", ['yes', 'no']],
        ]

    async def __call__(self):
        # Setup
        self.history.clear()
        self.header(self.title, h=0)
        self.record_narration(self.scenario, timestep=self.timestep)
        self.header(self.scenario, h=2)

        # Moves
        for move in range(self.moves):
            self.header('Move ' + str(move + 1), h=1)
            responses = snowglobe.History()
            for player in self.players:
                self.header('### ' + player.name, h=2)
                response = await player.respond(history=self.history)
                responses.add(player.name, response)
            self.header('### Result', h=2)
            outcome = await self.adjudicate(
                history=self.history, responses=responses, nature=self.nature,
                timestep=self.timestep, mode=self.mode)
            self.record_narration(outcome, timestep=self.timestep)

        # Answer questions
        for question in self.questions:
            self.header(question, h=1)
            await self.assess(history=self.history, query=question)
        for question, mc in self.mc_questions:
            self.header(question, h=1)
            await self.assess(history=self.history, query=question, mc=mc)


if __name__ == '__main__':
    sim = AzuristanCrimsonia()
    sim.run()
