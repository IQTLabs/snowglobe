#!/usr/bin/env python3

import fastapi

app = fastapi.FastAPI()

@app.get('/')
async def prompt():
    return {'message': 'testing 1 2 3'}

