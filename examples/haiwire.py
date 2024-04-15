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

import os
import csv
import random
import snowglobe

def load_csv(path):
    # Load a CSV file as a list of dictionaries
    datalist = []
    with open(path) as datafile:
        reader = csv.DictReader(datafile)
        for row in reader:
            datalist.append(row)
    return datalist


class HAIwire(snowglobe.Control):
    def __init__(self,
                 incident_path=os.path.join(os.path.dirname(__file__),
                                            'haiwire_files', 'incidents.csv'),
                 inject_path=os.path.join(os.path.dirname(__file__),
                                          'haiwire_files', 'injects.csv')):
        super().__init__()

        # Optionally truncate answers
        # self.llm.llm = self.llm.llm.bind(max_tokens=250)

        self.player = snowglobe.Player(
            llm=self.llm,
            name='Incident Response Team',
            persona='the management of a technology company')

        self.incidents = load_csv(incident_path)
        self.injects = load_csv(inject_path)

        self.query = 'What actions do you take in response?'
        self.partial_success_message = \
            'That response resolved the latest problem,'\
            ' but not the previous problems.'
        self.full_success_message = \
            'That response completely succeeded.'

        self.assessment_sets = [
            [
                "Yes or no, was a lawyer consulted at any point?",
            ], [
                "If these plans are implemented,"\
                " what is the most likely outcome?",
                "Yes or no, was this a good outcome to the situation?",
            ]
        ]

    def __call__(self):
        self.header('HAIWIRE', h=0)

        incident = random.choice(self.incidents)
        inject_count = {'GREEN':2, 'AMBER':3, 'RED':4}[incident['Card Color']]
        injects = random.sample(self.injects, inject_count)

        cards = [incident] + injects
        for i, card in enumerate(cards):
            self.header('"' + card['Card Title'].strip() + '"\n'
                        + card['Card Wording'].strip(), h=1)
            self.record_narration(card['Card Wording'].strip())
            if i > 0 and card['Card Wording'][-19:] == 'Draw next Incident!':
                break
            response = self.player.respond(
                history=self.history, query=self.query)
            self.record_response(self.player.name, response)
            if i > 0:
                d10roll = random.randint(1, 10)
                if d10roll >= 10:
                    self.header(self.full_success_message, h=1)
                    # self.record_narration(self.full_success_message)
                    break
                elif d10roll >= 8:
                    self.header(self.partial_success_message, h=1)
                    self.record_narration(self.partial_success_message)
        self.header('End of Incident', h=1)

        self.header('Discussion', h=0)
        for assessment_set in self.assessment_sets:
            set_history = snowglobe.History()
            for assessment in assessment_set:
                self.header(assessment, h=1)
                response = self.assess(
                    history=self.history,
                    responses=set_history,
                    query=assessment,
                    short=True)
                set_history.add('Question', assessment)
                set_history.add('Answer', response)


if __name__ == '__main__':
    sim = HAIwire()
    sim()
