#!/usr/bin/env python3

#   Copyright 2023-2024 IQT Labs LLC
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

import argparse
import snowglobe

class AzuristanCrimsonia(snowglobe.Control):
    def __init__(self, human):
        super().__init__()

        persona_azuristan_dove = 'the leader of Azuristan.  Your goal is to avoid war at all costs, and to preserve the sovereignty of Azuristan if possible.'
        persona_azuristan_hawk = 'the leader of Azuristan.  Your goal is to preserve the sovereignty of Azuristan, even if it requires starting a war.'
        persona_crimsonia_dove = 'the leader of Crimsonia.  Your goal is to avoid war at all costs, and to unify the Crimsonian people if possible.'
        persona_crimsonia_hawk = 'the leader of Crimsonia.  Your goal is to unify the Crimsonian people, even if it requires starting a war.'
        self.azuristan = snowglobe.Player(
            llm=self.llm,
            name='President of Azuristan',
            kind='human' if human >= 1 else 'ai',
            persona=persona_azuristan_dove)
        self.crimsonia = snowglobe.Player(
            llm=self.llm,
            name='Premier of Crimsonia',
            kind='human' if human >= 2 else 'ai',
            persona=persona_crimsonia_dove)

        self.background = """\
Azuristan and Crimsonia are neighboring countries in Central Asia.  Azuristan is a Western-backed democracy that suffers from endemic corruption.  Crimsonia is controlled by an autocratic government that stifles dissent and commits human rights violations.

Both countries have modern professional militaries, although Crimsonia's is slightly larger than Azuristan's.  In addition, Azuristan possesses ten nuclear weapons and Crimsonia has eight.

Most citizens of Azuristan are from the Azuristani ethnic group, and most citizens of Crimsonia are from the Crimsonian ethnic group.  The only exception is Azuristan's province of Tyriana, on the border with Crimsonia.  Most residents of Tyriana belong to the Crimsonian ethnic group.

The animosity between Azuristan and Crimsonia extends back over centuries of ethnic tension and intermittent warfare.  Recent years have been fairly calm.  However, that suddenly changes when Tyriana declares independence.  Amidst the crisis, local leaders in Tyriana ask Crimsonia to come to the province's defense, and the same leaders indicate that they want Tyriana to become part of Crimsonia."""

        self.questions = [
            'What was the final status of Tyriana?  Your answer should be "part of Azuristan" or "part of Crimsonia" or "independent" or "not yet determined".',
            'Did armed conflict occur?  Your answer should be "yes" or "no".',
        ]

        self.moves = 3
        self.timestep = 'month'
        self.nature = True

    def __call__(self):
        self.header('Azuristan and Crimsonia', h=0)
        self.header(self.background, h=2)
        self.record_narration(self.background, timestep=self.timestep)

        for move in range(self.moves):
            self.header('Move ' + str(move + 1), h=1)
            responses = snowglobe.History()
            self.header('### Azuristan', h=2)
            a_response = self.azuristan.respond(history=self.history)
            responses.add(self.azuristan.name, a_response)
            self.header('### Crimsonia', h=2)
            c_response = self.crimsonia.respond(history=self.history)
            responses.add(self.crimsonia.name, c_response)
            self.header('### Result', h=2)
            r_response = self.adjudicate(
                history=self.history, responses=responses,
                nature=self.nature, timestep=self.timestep)
            self.record_narration(r_response, timestep=self.timestep)

        self.header('Assessment', h=0)
        self.header('History', h=1)
        print(self.history.str())
        for question in self.questions:
            self.header(question, h=1)
            self.assess(history=self.history, query=question)
        self.chat()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', action='store', default=0, type=int,
                        choices=[0, 1, 2], help='Number of human players')
    args = parser.parse_args()

    sim = AzuristanCrimsonia(human=args.human)
    sim()
