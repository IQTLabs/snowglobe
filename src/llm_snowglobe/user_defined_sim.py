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
import argparse
import logging

from llm_snowglobe import UserDefinedGame
from llm_snowglobe.core import Configuration 

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
  parser.add_argument(
    '-r', '--runs', help='number of times to run the game', action='store',
    default=5, type=int)
  parser.add_argument(
    '-n', '--simulation-name', help='name of the simulation (Used to name data files)', action='store',
    default='simulation')
  parser.add_argument(
    '-s', '--simulation-mode', help='Whether to run the games as a simualtion', action='store_true',
    default=False)

  args = parser.parse_args()

  logger = configure_logging(args.log_file)
  config = Configuration(args.config_file)
  logger.debug(f"Config loaded from {args.config_file}")
  logger.info(f"Running scenario {config.title} {args.runs} times")
  if config:
    for i in range(args.runs):
      sim = UserDefinedGame(config, logger, args.simulation_mode, args.simulation_name, i)
      # sim.run()
      # sim.header('End of Simulation %i' % (i + 1,), h=2)
      # print(answer_log)
      sim.run()

if __name__ == '__main__':
  main()