#!/usr/bin/env python3

import fastapi
import fastapi.staticfiles

app = fastapi.FastAPI()

@app.get('/data')
async def prompt():
    return {'message': 'testing 1 2 3'}

app.mount('/', fastapi.staticfiles.StaticFiles(directory='terminal', html=True), name='terminal')
