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
import time
import threading
import logging
import asyncio
import argparse

from llm_snowglobe.core import Configuration, Control, Database, History, Player

class TerminateChats(Exception):
  def __init__(self):
    super().__init__()

class UserDefinedGame(Control):
  def __init__(self, config, logger, simulation_mode=False, simulation_name='simulation', run_num=0):
    self.logger = logger
    self.verbosity = 2

    self.goals = dict() 
    for goal_name in config.goals:
      self.goals[goal_name] = config.goals[goal_name]

    if simulation_mode:
      self.ioid = f"{simulation_name}_{run_num}"
    else:
      self.ioid = uuid.uuid4().hex

    gameroom = f"game{self.ioid}"
    chatroom = f"chat{self.ioid}"
    with open(config.game_id_file,'w') as gif:
      gif.write(self.ioid)

    self.game_id_file = config.game_id_file
    self.data_dir = config.data_dir
    self.db = Database(self.ioid, path=config.data_dir, initialize=True)
    self.name = 'Game Control'
    super().__init__(database=self.db, verbosity=self.verbosity, logger=self.logger, name=self.name, ioid=self.ioid)

    self.title = config.title
    self.players = list()
    self.advisors = list()

    for name in config.players:
      cfg_player = config.players[name]
      sg_player = None

      if cfg_player['kind'] == 'human': #player is human
        ioid = uuid.uuid4()
        if cfg_player['ioid']:
          ioid = cfg_player['ioid']

        infodocs = []
        if 'infodocs' in cfg_player:
          infodocs = cfg_player['infodocs']
        sg_player = Player(
          database=self.db,
          verbosity = self.verbosity,
          llm=self.llm,
          name=name,
          kind=cfg_player['kind'],
          ioid=ioid,
          iodict={
            'chatrooms': [gameroom, chatroom],
            'infodocs': infodocs,
        })
        sg_player.gameroom = gameroom
        sg_player.chatroom = chatroom
      elif cfg_player['kind'] == 'ai': #player is ai
        persona = cfg_player['persona']
        persona_goals = list()
        for g in cfg_player['goals']:
          persona_goals.append(self.goals[g])
        sg_player = Player(
          database=self.db,
          verbosity = self.verbosity,
          llm=self.llm,
          name=name,
          persona=f"{persona}.  {persona_goals}"
        )
      else:
        self.logger.warning(f"Skipping player {name} with invalid type {cfg_player['kind']}")

      if sg_player:
        self.logger.info(f"Adding {cfg_player['kind']} player {name} to game {self.ioid}")
        self.players.append(sg_player)

    for name in config.advisors:
      cfg_advisor = config.advisors[name]
      persona = cfg_advisor['persona']
      persona_goals = list()
      for g in cfg_advisor['goals']:
        persona_goals.append(self.goals[g])
      sg_advisor = Player(
        database=self.db,
        verbosity = self.verbosity,
        llm=self.llm,
        name=name,
        persona=f"{persona}.  {persona_goals}")
      sg_advisor.chatroom = chatroom
      self.logger.info(f"Adding AI advisor {name} to game {self.ioid}")
      self.advisors.append(sg_advisor)
    
    self.scenario = config.scenario
    self.moves = config.moves
    self.timestep = config.timestep
    self.nature = config.nature
    self.mode = config.mode


    # User interface properties
    self.logger.info(f"Setting interface properties")
    prop = self.db.add_property
    for player in self.players:
      if player.kind == 'human':
        prop(player.gameroom, 'title', 'Play the Game')
        prop(player.gameroom, 'instruction', 'Enter your response.')
        prop(player.chatroom, 'title', 'Your AI Advisor')
        prop(player.chatroom, 'instruction', 'Ask your AI advisor.')
    prop('ac_game_help', 'title', 'Help')
    prop('ac_game_help', 'content', "## Help\n\nClick *Play the Game* to enter your response for each move, or click *Your AI Advisor* to consult with your AI advisor about what to do.")
    prop('ac_game_help', 'format', 'markdown')
    self.logger.info(f"Commiting to db at {self.db.path}")
    self.db.commit()

  async def game(self):
    # Setup
    self.history.clear()
    self.logger.info("History cleared.")
    self.header(self.title, h=0)
    self.record_narration(self.scenario, timestep=self.timestep)
    self.header(self.scenario, h=2)
    self.logger.info("Interface headers set.")
    for player in self.players:
      if player.kind == 'human':
        self.interface_send_message(
          player.gameroom, 'You are {}.'.format(
            player.name), 'markdown')

    # Moves
    for move in range(self.moves):
      self.logger.info(f"Taking move number {move}")
      self.header('Move ' + str(move + 1), h=1)
      responses = History()
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
          self.logger.info(f"Response received from AI player {player.name}")
        responses.add(player.name, response)
      self.header('### Result', h=2)
      self.logger.info(f"Adjudicating results for move {move} of game {self.ioid}")
      outcome = await self.adjudicate(
        history=self.history, responses=responses, nature=self.nature,
        timestep=self.timestep, mode=self.mode)
      self.logger.info(f"Results for move {move} of game {self.ioid} successfully adjudicated. Recording narration.")
      self.record_narration(outcome, timestep=self.timestep)

    # Conclusion
    for player in self.players:
      if player.kind == 'human':
        content = await self.history[-1].textonly() \
          + '\n\n**Game Over**'
        self.interface_send_message(
          player.gameroom, content, 'markdown')
        self.logger.info(f"Game {self.ioid} complete.")

    self.logger.info("Terminating advisor chats.")

    for advisor in self.advisors:
      advisor.active = False

  async def __call__(self):
    # try:
    async with asyncio.TaskGroup() as group:
      for advisor in self.advisors:
        group.create_task(advisor.chat_session(advisor.chatroom, history=self.history))
      group.create_task(self.game())
    # except* TerminateChats:
    #   self.logger.info("Terminate chats exception caught")
    #   time.sleep(3)
      
     
    # self.tasks.append()
    # await asyncio.gather(*self.tasks)
    # await asyncio.gather(*[advisor.chat_session(advisor.chatroom,
    #                                             history=self.history)
    #                        for advisor in self.advisors], self.game())


def configure_logging(log_file):
  logger = logging.getLogger(__name__)
  logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
  logger.info('Logging started')
  return logger

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-c','--config-file', help='location of a yaml config file', action='store',
    default='/config/game.yaml')
  parser.add_argument(
    '-l','--log-file', help='location of a file to log to', action='store',
    default='/home/snowglobe/logs/snowglobe.log')
  args = parser.parse_args()

  logger = configure_logging(args.log_file)
  config = Configuration(args.config_file)
  logger.debug(f"Config loaded from {args.config_file}")
  if config:
    sim = UserDefinedGame(config, logger)
    sim.run()

if __name__ == '__main__':
  main()
  