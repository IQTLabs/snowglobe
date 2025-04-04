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
import time
import asyncio
import watchfiles
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
        json.dump(globals()['databank'], f, indent=4)

app.on_startup(load_databank)
app.on_shutdown(save_databank)
app.on_exception(save_databank)

app.add_static_file(url_path='/ai.png', local_file=os.path.join(
    here, 'assets/ai.png'))
app.add_static_file(url_path='/human.png', local_file=os.path.join(
    here, 'assets/human.png'))

@ui.page('/')
async def interface_page():

    async def load_id(idval):
        if len(idval) == 0:
            ui.notify('Enter your ID.')
        elif not idval in databank['players']:
            ui.notify('ID not found.')
        else:
            app.storage.tab['id'] = idval
            app.storage.tab['logged_in'] = True
            app.storage.tab['message_count'] = 0
            login_name.text = databank['players'][idval]['name']
            display_messages.refresh()
            display_infodoc.refresh()

    async def send_message():
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        message = {
            'text': chattext.value,
            'name': databank['players'][idval]['name'],
            'stamp': time.ctime(),
            'avatar': 'human.png',
        }
        chatroom = databank['chatrooms'][databank['players'][idval]['chatrooms'][0]]
        if not 'log' in chatroom:
            chatroom['log'] = []
        chatroom['log'].append(message)
        display_messages.refresh()
        chattext.set_value('')
        save_databank()

    async def get_disk_updates():
        while True:
            async for changes in watchfiles.awatch(datapath):
                break
            load_databank()
            display_messages.refresh()
            display_infodoc.refresh()

    @ui.refreshable
    def display_messages():
        if 'id' in app.storage.tab:
            idval = app.storage.tab['id']
            name = databank['players'][idval]['name']
            chatroom = databank['chatrooms'][databank['players'][idval]['chatrooms'][0]]
            for message in chatroom['log']:
                ui.chat_message(sent=name == message['name'], **message).classes('w-full')
            if len(chatroom['log']) > app.storage.tab['message_count']:
                message_window.scroll_to(percent=100)
                app.storage.tab['message_count'] = len(chatroom['log'])

    @ui.refreshable
    def display_infodoc():
        if 'id' in app.storage.tab:
            idval = app.storage.tab['id']
            infodoc = databank['infodocs'][databank['players'][idval]['infodocs'][0]]
            if 'format' not in infodoc or infodoc['format'] == 'plaintext':
                infocontent = ui.label(infodoc['content'])
            elif infodoc['format'] == 'markdown':
                infocontent = ui.markdown(infodoc['content'].replace('\\n', chr(10)))
            elif infodoc['format'] == 'html':
                infocontent = ui.html(infodoc['content'])

    await ui.context.client.connected()
    app.storage.tab['logged_in'] = False
    ui.timer(0, get_disk_updates, once=True)

    with ui.left_drawer(top_corner=True, bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image(os.path.join(here, 'assets/snowglobe.png')).props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in', backward=lambda x: not x):
                login_id = ui.input('ID', placeholder='Player ID #').props('size=8')
                ui.button('Connect', on_click=lambda x: load_id(login_id.value))
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in'):
                login_numb = ui.label('ID').bind_text_from(app.storage.tab, 'id')
                login_name = ui.label('Name')
    with ui.header().style('background-color: #B4C7E7'):
        with ui.tabs().classes('w-full') as tabs:
            chattab = ui.tab('Chat')
            infotab = ui.tab('Info')
    with ui.tab_panels(tabs, value=chattab).classes('absolute-full'):
        with ui.tab_panel(chattab).classes('h-full'):
            with ui.column().classes('w-full items-center h-full'):
                with ui.scroll_area().classes('w-full h-full border') as message_window:
                    display_messages()
                chattext = ui.textarea(placeholder='Ask the AI assistant.').classes('w-full border').style('height: auto; padding: 0px 5px')
                ui.button('Send', on_click=send_message)
                ui.label('Do not submit sensitive or personal information.').style('font-size: 10px')
        with ui.tab_panel(infotab):
            with ui.scroll_area().classes('w-full h-full'):
                display_infodoc()


def snowglobe_interface(host='0.0.0.0', port=8000):
    ui.run(host=host, port=port, title='Snow Globe User Interface',
           favicon=os.path.join(here, 'terminal/favicon.ico'),
           reload=True)


if __name__ in {'__main__', '__mp_main__'}:
    snowglobe_interface()
