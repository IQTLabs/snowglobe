#!/usr/bin/env python3

import fastapi
import fastapi.staticfiles
import os
import json

app = fastapi.FastAPI()

@app.get('/prompt/{label}/{count}')
async def prompt(label: int, count: int):
    base_path = 'messages'
    path = os.path.join(base_path, '%i_%02i_prompt.json' % (label, count))
    with open(path, 'r') as f:
        j = json.load(f)
    return j

app.mount('/', fastapi.staticfiles.StaticFiles(directory='terminal', html=True), name='terminal')
