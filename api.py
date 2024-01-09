#!/usr/bin/env python3

import os
import json
import fastapi
import fastapi.staticfiles
from pydantic import BaseModel

app = fastapi.FastAPI()
here = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(here, 'messages')
term_path = os.path.join(here, 'terminal')

class Answer(BaseModel):
    content: str

@app.get('/prompt/{label}/{count}')
async def prompt(label: int, count: int):
    path = os.path.join(base_path, '%i_%i_prompt.json' % (label, count))
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        j = json.load(f)
    return j

@app.post('/answer/{label}/{count}')
async def answer(label: int, count: int, answer: Answer):
    path = os.path.join(base_path, '%i_%i_answer.json' % (label, count))
    with open(path, 'w') as f:
        json.dump(answer.dict(), f)
    return 0

app.mount('/', fastapi.staticfiles.StaticFiles(directory=term_path, html=True), name='terminal')
