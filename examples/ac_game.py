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

import random
import asyncio
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
        ioid = str(random.randint(100000, 999999)) # Unique for each human
        gameroom = 'game' + ioid
        chatroom = 'chat' + ioid

        self.title = 'Azuristan and Crimsonia'
        self.players = [
            snowglobe.Player(
                llm=self.llm,
                name='President of Azuristan',
                kind='human',
                ioid=ioid,
                iodict={
                    'chatrooms': [gameroom, chatroom],
                    'infodocs': ['ac_game_help'],
                }),
            snowglobe.Player(
                llm=self.llm,
                name='Premier of Crimsonia',
                persona='the leader of Crimsonia.  {}'.format(goals[
                    'crimsonia_dove'])),
        ]
        self.advisors = [
            snowglobe.Player(
                llm=self.llm,
                name='Advisor to the President of Azuristan',
                persona='an advisor to the leader of Azuristan.  {}'.format(
                    goals['azuristan_dove'])),
        ]
        self.players[0].gameroom = gameroom
        self.players[0].chatroom = chatroom
        self.advisors[0].chatroom = chatroom
        self.scenario = """\
Azuristan and Crimsonia are neighboring countries in Central Asia.  Azuristan is a Western-backed democracy that suffers from endemic corruption.  Crimsonia is controlled by an autocratic government that stifles dissent and commits human rights violations.

Both countries have modern professional militaries, although Crimsonia's is slightly larger than Azuristan's.  In addition, Azuristan possesses ten nuclear weapons and Crimsonia has eight.

Most citizens of Azuristan are from the Azuristani ethnic group, and most citizens of Crimsonia are from the Crimsonian ethnic group.  The only exception is Azuristan's province of Tyriana, on the border with Crimsonia.  Most residents of Tyriana belong to the Crimsonian ethnic group.

The animosity between Azuristan and Crimsonia extends back over centuries of ethnic tension and intermittent warfare.  Recent years have been fairly calm.  However, that suddenly changes when Tyriana declares independence.  Amidst the crisis, local leaders in Tyriana ask Crimsonia to come to the province's defense, and the same leaders indicate that they want Tyriana to become part of Crimsonia."""
        self.moves = 3
        self.timestep = 'month'
        self.nature = True
        self.mode = ['geopol']

        # User interface properties
        prop = snowglobe.db.add_property
        for player in self.players:
            if player.kind == 'human':
                prop(player.gameroom, 'title', 'Play the Game')
                prop(player.gameroom, 'instruction', 'Enter your response.')
                prop(player.chatroom, 'title', 'Your AI Advisor')
                prop(player.chatroom, 'instruction', 'Ask your AI advisor.')
        prop('ac_game_help', 'title', 'Help')
        prop('ac_game_help', 'content', "## Help\n\nClick *Play the Game* to enter your response for each move, or click *Your AI Advisor* to consult with your AI advisor about what to do.")
        prop('ac_game_help', 'format', 'markdown')
        snowglobe.db.commit()

    async def game(self):
        # Setup
        self.history.clear()
        self.header(self.title, h=0)
        self.record_narration(self.scenario, timestep=self.timestep)
        self.header(self.scenario, h=2)
        for player in self.players:
            if player.kind == 'human':
                self.interface_send_message(
                    player.gameroom, 'You are {}.'.format(
                        player.name), 'markdown')

        # Moves
        for move in range(self.moves):
            self.header('Move ' + str(move + 1), h=1)
            responses = snowglobe.History()
            for player in self.players:
                self.header('### ' + player.name, h=2)
                if player.kind == 'human':
                    content = await self.history[-1].textonly() \
                        + '\n\n**How do you respond?**'
                    self.interface_send_message(
                        player.gameroom, content, 'markdown')
                    response = player.interface_get_message(player.gameroom)
                else:
                    response = await player.respond(history=self.history)
                responses.add(player.name, response)
            self.header('### Result', h=2)
            outcome = await self.adjudicate(
                history=self.history, responses=responses, nature=self.nature,
                timestep=self.timestep, mode=self.mode)
            self.record_narration(outcome, timestep=self.timestep)

        # Conclusion
        for player in self.players:
            if player.kind == 'human':
                content = await self.history[-1].textonly() \
                    + '\n\n**Game Over**'
                self.interface_send_message(
                    player.gameroom, content, 'markdown')

    async def __call__(self):
        await asyncio.gather(*[advisor.chat_session(advisor.chatroom,
                                                    history=self.history)
                               for advisor in self.advisors], self.game())


if __name__ == '__main__':
    sim = AzuristanCrimsonia()
    sim.run()
