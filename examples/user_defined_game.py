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

import uuid
import logging
import asyncio
import llm_snowglobe as snowglobe

from ruamel.yaml import YAML


class UserDefinedGame(snowglobe.Control):
    def __init__(self, config, logger):
        super().__init__()

        self.logger = logger
        goals = dict() 
        for goal_name in config['goals']:
            goals[goal_name] = config['goals'][goal_name]

        ioid = config['ioid'] if 'ioid' in config else uuid.uuid4()
        gameroom = f"game{ioid}"
        chatroom = f"chat{ioid}"

        self.title = config['title']
        self.players = list()
        self.advisors = list()

        for name in config['players']:
            cfg_player = config['players'][name]
            sg_player = None

            if cfg_player['kind'] == 'human': #player is human
                ioid = uuid.uuid4()
                if cfg_player['ioid']:
                    ioid = cfg_player['ioid']

                infodocs = []
                if 'infodocs' in cfg_player:
                    infodocs = cfg_player['infodocs']
                sg_player = snowglobe.Player(
                    llm=self.llm,
                    name=name,
                    kind=cfg_player['kind'],
                    ioid=ioid,
                    iodict={
                        'chatrooms': [gameroom, chatroom],
                        'infodocs': ['ac_game_help'],
                })
                sg_player.gameroom = gameroom
                sg_player.chatroom = chatroom
            elif cfg_player['kind'] == 'ai': #player is ai
                persona = cfg_player['persona']
                persona_goals = list()
                for g in cfg_player['goals']:
                    persona_goals.append(goals[g])
                sg_player = snowglobe.Player(
                    llm=self.llm,
                    name=name,
                    persona=f"{persona}.  {persona_goals}"
                )
            else:
                self.logger.warning(f"Skipping player {name} with invalid type {cfg_player['kind']}")

            if sg_player:
                self.logger.info(f"Adding {cfg_player['kind']} player {name} to game {ioid}")
                self.players.append(sg_player)

        for name in config['advisors']:
            cfg_advisor = config['advisors'][name]
            persona = cfg_advisor['persona']
            persona_goals = list()
            for g in cfg_advisor['goals']:
                persona_goals.append(goals[g])
            sg_advisor = snowglobe.Player(
                llm=self.llm,
                name=name,
                persona=f"{persona}.  {persona_goals}")
            sg_advisor.chatroom = chatroom
            self.logger.info(f"Adding AI advisor {name} to game {ioid}")
            self.advisors.append(sg_advisor)
        
        self.scenario = config['scenario']
        self.moves = config['moves']
        self.timestep = config['timestep']
        self.nature = config['nature']
        self.mode = config['mode']


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
        self.logger.info(f"History cleared.")
        self.header(self.title, h=0)
        self.record_narration(self.scenario, timestep=self.timestep)
        self.header(self.scenario, h=2)
        self.logger.info(f"Interface headers set.")
        for player in self.players:
            if player.kind == 'human':
                self.interface_send_message(
                    player.gameroom, 'You are {}.'.format(
                        player.name), 'markdown')

        # Moves
        for move in range(self.moves):
            self.logger.info(f"Taking move number {move}")
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
                    self.logger.info(f"Response received from player {player.name} in {player.gameroom}")
                else:
                    response = await player.respond(history=self.history)
                responses.add(player.name, response)
            self.header('### Result', h=2)
            self.logger.info(f"Adjudicating results for move {move} of game {self.ioid}")
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

def load_config(cfg_file):
    config = None
    file_config = None
    with open(cfg_file, "r") as cfg_file:
      yaml = YAML(typ="safe")
      file_config = yaml.load(cfg_file)

    if file_config:
        config = {}
        config['goals'] = file_config['goals']
        
        game_id = uuid.uuid4()
        if file_config['gameroom']:
            config['gameroom'] = file_config['gameroom']
        else:
            config['gameroom'] = 'game' + game_id

        if file_config['chatroom']:
            config['chatroom'] = file_config['chatroom']
        else:
            config['chatroom'] = 'chat' + game_id

        config['title'] = file_config['title']
        config['scenario'] = file_config['scenario']
        config['moves'] = file_config['moves']
        config['timestep'] = file_config['timestep']
        config['nature'] = file_config['nature']
        config['mode'] = file_config['mode']
        config['players'] = dict()
        config['advisors'] = dict()

        for player in file_config['players']:
            config['players'][player] = file_config['players'][player]

        for advisor in file_config['advisors']:
            config['advisors'][advisor] = file_config['advisors'][advisor]

    return config

def configure_logging(log_file):
  logger = logging.getLogger(__name__)
  logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
  logger.info('Logging started')
  return logger

if __name__ == '__main__':
    logger = configure_logging('logs/snowglobe.log')

    config_file = '/config/game.yaml'
    config = load_config(config_file)
    logger.debug(f"Config loaded from {config_file}")
    if config:
        sim = UserDefinedGame(config, logger)
        sim.run()
