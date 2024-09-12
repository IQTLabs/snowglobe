#!/usr/bin/env python3

#   Copyright 2024 IQT Labs LLC
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
import json
import fastapi
import fastapi.staticfiles
from pydantic import BaseModel
import platformdirs

app = fastapi.FastAPI()
here = os.path.dirname(os.path.abspath(__file__))
base_path = platformdirs.user_data_dir('snowglobe')
term_path = os.path.join(here, 'terminal')

class Answer(BaseModel):
    content: str

@app.get('/prompt/{label}/{count}')
async def prompt(label: int, count: int):
    path = os.path.join(
        base_path, str(label), '%i_%i_prompt.json' % (label, count))
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

@app.post('/answer/{label}/{count}')
async def answer(label: int, count: int, answer: Answer):
    path = os.path.join(
        base_path, str(label), '%i_%i_answer.json' % (label, count))
    if os.path.exists(os.path.dirname(path)):
        with open(path, 'w') as f:
            json.dump(answer.dict(), f)
    else:
        print('Unexpected API response [ID %i # %i]' % (label, count))

app.mount('/', fastapi.staticfiles.StaticFiles(directory=term_path, html=True), name='terminal')
