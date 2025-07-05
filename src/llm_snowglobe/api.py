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
import pydantic
from llm_snowglobe import db, default_chatroom

app = fastapi.FastAPI()
here = os.path.dirname(os.path.abspath(__file__))
term_path = os.path.join(here, 'terminal')

class Message(pydantic.BaseModel):
    content: str
    format: str
    name: str
    stamp: str
    avatar: str

@app.get('/read/{ioid}/{count}')
async def prompt(ioid: str, count: int):
    if count == 0:
        name = db.get_name(ioid)
        if name is None:
            return {}
        else:
            return {'name': name}
    else:
        chatroom = default_chatroom(ioid)
        chatlog = db.get_chatlog(chatroom)
        if len(chatlog) >= count:
            return  chatlog[count - 1]
        else:
            return {}

@app.post('/post/{ioid}')
async def answer(ioid: str, answer: Message):
    chatroom = default_chatroom(ioid)
    message = answer.dict()
    db.send_message(chatroom, **message)
    db.commit()

app.mount('/', fastapi.staticfiles.StaticFiles(directory=term_path, html=True), name='terminal')
