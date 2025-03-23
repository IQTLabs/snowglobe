#!/usr/bin/env python3

#   Copyright 2025 IQT Labs LLC
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
import asyncio
from nicegui import ui, app

here = os.path.dirname(os.path.abspath(__file__))
datapath = 'databank.json'
databank = None

def load_databank():
    if os.path.exists(datapath):
        with open(datapath, 'r') as f:
            globals()['databank'] = json.load(f)
    else:
        globals()['databank'] = {'players': {}, 'chatrooms': {},
                                 'infodocs': {}, 'editdocs': {}}

def save_databank():
    with open(datapath, 'w') as f:
        json.dump(globals()['databank'], f)

app.on_startup(load_databank)
app.on_shutdown(save_databank)
app.on_exception(save_databank)


@ui.page('/')
async def interface_page():

    async def load_id(idval):
        if len(idval) > 0:
            app.storage.tab['id'] = idval
            app.storage.tab['logged_in'] = True
            login_name.text = databank['players'][idval]['name']

    await ui.context.client.connected()
    app.storage.tab['logged_in'] = False
    with ui.left_drawer(bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image(os.path.join(here, 'terminal/snowglobe.png')).props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in', backward=lambda x: not x):
                login_id = ui.input('ID', placeholder='Player ID #').props('size=8')
                ui.button('Connect', on_click=lambda x: load_id(login_id.value))
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in'):
                login_numb = ui.label('ID').bind_text_from(app.storage.tab, 'id')
                login_name = ui.label('Error: ID Not Found')
    with ui.tabs().classes('w-full') as tabs:
        chattab = ui.tab('Chat')
        infotab = ui.tab('Info')
    ui.separator()
    with ui.tab_panels(tabs, value=chattab).classes('w-full'):
        with ui.tab_panel(chattab).classes(''):
            with ui.column().classes('w-full items-center'):
                ui.scroll_area().classes('w-full h-[50vh] border')
                ui.textarea(placeholder='Ask the AI assistant.').classes('w-full border').style('height: auto; padding: 0px 5px')
                ui.button('Send', on_click=lambda: ui.notify('Click!'))
                ui.label('Do not submit sensitive or personal information.').style('font-size: 10px')
        with ui.tab_panel(infotab):
            ui.label('Information')


def snowglobe_interface(host='0.0.0.0', port=8000):
    ui.run(host=host, port=port, title='Snow Globe User Interface',
           favicon=os.path.join(here, 'terminal/favicon.ico'),
           reload=True)


if __name__ in {'__main__', '__mp_main__'}:
    snowglobe_interface()
