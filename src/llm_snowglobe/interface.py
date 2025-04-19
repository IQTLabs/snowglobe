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
import markdown2
import watchfiles
from nicegui import ui, app, events

here = os.path.dirname(os.path.abspath(__file__))
datapath = 'databank.json'
databank = None
datastep = 0

async def load_databank():
    if not os.path.exists(datapath):
        globals()['databank'] = {'players': {}, 'chatrooms': {},
                                 'infodocs': {}, 'editdocs': {}}
        save_databank()
    while True:
        while True:
            try:
                with open(datapath, 'r') as f:
                    globals()['databank'] = json.load(f)
            except json.decoder.JSONDecodeError:
                print('! Databank load error')
                time.sleep(0.1)
            else:
                break
        globals()['datastep'] += 1
        async for changes in watchfiles.awatch(datapath):
            break

def save_databank():
    with open(datapath, 'w') as f:
        json.dump(databank, f, indent=4)

app.on_startup(load_databank)

app.add_static_file(url_path='/ai.png', local_file=os.path.join(
    here, 'assets/ai.png'))
app.add_static_file(url_path='/human.png', local_file=os.path.join(
    here, 'assets/human.png'))

@ui.page('/')
async def interface_page():

    async def set_id(idval):
        if len(idval) == 0:
            ui.notify('Enter your ID.')
        elif not idval in databank['players']:
            ui.notify('ID not found.')
        else:
            app.storage.tab['id'] = idval
            app.storage.tab['message_count'] = 0
            login_numb.text = app.storage.tab['id']
            login_name.text = databank['players'][idval]['name']
            preloginrow.set_visibility(False)
            postloginrow.set_visibility(True)
            setup_tabs.refresh()
            setup_tab_panels.refresh()
            #await display_all()

    @ui.refreshable
    def setup_tabs():
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        default_title = {
            'chatrooms': 'Chat',
            'infodocs': 'Info',
            'editdocs': 'Edit',
        }
        for resource_type in default_title:
            if resource_type in databank['players'][idval]:
                for resource in databank['players'][idval][resource_type]:
                    tabvars[resource] = {}
                    tabvars[resource]['tab'] = ui.tab(resource) #change to title

    @ui.refreshable
    def setup_tab_panels():
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        setup_func = {
            'chatrooms': setup_chatroom,
            'infodocs': setup_infodoc,
            'editdocs': setup_editdoc,
        }
        for resource_type in setup_func:
            if resource_type in databank['players'][idval]:
                for resource in databank['players'][idval][resource_type]:
                    setup_func[resource_type].refresh(resource)

    @ui.refreshable
    def setup_chatroom(resource):
        with ui.tab_panel(tabvars[resource]['tab']).classes('h-full'):
            with ui.column().classes('w-full items-center h-full'):
                with ui.scroll_area().classes('w-full h-full border') as tabvars[resource]['message_window']:
                    display_messages(resource)
                    ui.notify('setup_chat')
                tabvars[resource]['chattext'] = ui.textarea(placeholder='Ask the AI assistant.').classes('w-full border').style('height: auto; padding: 0px 5px')
                ui.button('Send', on_click=lambda: send_message(resource)) ##
                ui.label('Do not send sensitive or personal information.').style('font-size: 10px')

    @ui.refreshable
    def setup_infodoc(resource):
        with ui.tab_panel(tabvars[resource]['tab']).classes('absolute-full'):
            with ui.scroll_area().classes('w-full h-full absolute-full'):
                display_infodoc(resource)

    @ui.refreshable
    def setup_editdoc(resource):
        with ui.tab_panel(tabvars[resource]['tab']).classes('absolute-full'):
            with ui.column().classes('w-full items-center h-full'):
                tabvars[resource]['editobj'] = ui.textarea().classes('w-full').props('input-class=h-80')
                display_editdoc(resource)
                ui.button('Submit', on_click=lambda: submit_editdoc(resource)) ##
                ui.label('Do not input sensitive or personal information.').style('font-size: 10px')


    async def display_all():
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        display_func = {
            'chatrooms': display_messages,
            'infodocs': display_infodoc,
        }
        for resource_type in display_func:
            if resource_type in databank['players'][idval]:
                for resource in databank['players'][idval][resource_type]:
                    display_func[resource_type].refresh(resource)

    @ui.refreshable
    def display_messages(resource):
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        name = databank['players'][idval]['name']
        chatroom = databank['chatrooms'][resource]
        if not 'log' in chatroom:
            return
        for message in chatroom['log']:
            if 'format' not in message or message['format'] == 'plaintext':
                text = message['content']
                text_html = False
            elif message['format'] == 'markdown':
                text = markdown2.markdown(message['content'])
                text_html = True
            elif message['format'] == 'html':
                text = message['content']
                text_html = True
            sent = message['name'] == name
            ui.chat_message(text=text,
                            name=message['name'],
                            stamp=message['stamp'],
                            avatar=message['avatar'],
                            sent=sent,
                            text_html=text_html,
                            ).classes('w-full')
        if len(chatroom['log']) > app.storage.tab['message_count']:
            tabvars[resource]['message_window'].scroll_to(percent=100)
            app.storage.tab['message_count'] = len(chatroom['log'])

    @ui.refreshable
    def display_infodoc(resource):
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        infodoc = databank['infodocs'][resource]
        if 'format' not in infodoc or infodoc['format'] == 'plaintext':
            ui.label(infodoc['content']).classes('w-full h-full')
        elif infodoc['format'] == 'markdown':
            ui.markdown(infodoc['content'].replace('\\n', chr(10))).classes('w-full h-full')
        elif infodoc['format'] == 'html':
            ui.html(infodoc['content']).classes('w-full h-full')

    @ui.refreshable
    def display_editdoc(resource):
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        editobj = tabvars[resource]['editobj']
        editobj.bind_value(app.storage.general, resource)
        if 'readonly' in databank['editdocs'][resource]:
            if databank['players'][idval]['name'] in \
               databank['editdocs'][resource]['readonly']:
                editobj.enabled = False

        # Code to reposition cursor, which by default gets moved to
        # the end of the textarea if someone else modifies the text.
        handler_setup = '''
        if (typeof window.editdoccursor == 'undefined') {
            window.editdoccursor = {};
        }
        if (typeof window.editdoccursor.%s == 'undefined') {
            window.editdoccursor.%s = {};
        }
        info = window.editdoccursor.%s;
        info.value = '';
        info.selectionStart = 0;
        info.selectionEnd = 0;
        info.selfChange = false;
        ''' % tuple([editdocname] * 3)
        text_change_handler = '''() => {
            window.editdoccursor.%s.selfChange = true;
        }''' % editdocname
        cursor_move_handler = '''() => {
            const element = getElement(%s).$refs.qRef.getNativeElement();
            const info = window.editdoccursor.%s;
            // Relocate cursor if text was changed but not by this user
            if (element.value !== info.value && !info.selfChange) {
                var newStart = element.selectionStart;
                var newEnd = element.selectionEnd;
                // Relocate selectionStart
                if (element.value.substring(info.selectionStart + element.value.length - info.value.length) == info.value.substring(info.selectionStart)) {
                    newStart = info.selectionStart + element.value.length - info.value.length;
                } else if (element.value.substring(0, info.selectionStart) == info.value.substring(0, info.selectionStart)) {
                    newStart = info.selectionStart;
                }
                // Relocate selectionEnd
                if (element.value.substring(info.selectionEnd + element.value.length - info.value.length) == info.value.substring(info.selectionEnd)) {
                    newEnd = info.selectionEnd + element.value.length - info.value.length;
                } else if (element.value.substring(0, info.selectionEnd) == info.value.substring(0, info.selectionEnd)) {
                    newEnd = info.selectionEnd;
                }
                // Implement any specified relocations
                if (newStart !== element.selectionStart || newEnd !== element.selectionEnd) {
                    element.setSelectionRange(newStart, newEnd);
                }
            }
            // Update record of previous cursor location
            info.value = element.value;
            info.selectionStart = element.selectionStart;
            info.selectionEnd = element.selectionEnd;
            info.selfChange = false;
        }''' % (editobj.id, editdocname)
        ui.run_javascript(handler_setup)
        editobj.on('update:model-value', js_handler=text_change_handler)
        editobj.on('selectionchange', js_handler=cursor_move_handler)
        #editobj.on('update:model-value', lambda: ui.notify('text_change'))
        #editobj.on('selectionchange', lambda: ui.notify('cursor_move'))

    async def send_message(resource):
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        chattext = tabvars[resource]['chattext']
        message = {
            'content': chattext.value,
            'format': 'plaintext',
            'name': databank['players'][idval]['name'],
            'stamp': time.ctime(),
            'avatar': 'human.png',
        }
        chatroom = databank['chatrooms'][resource]
        if not 'log' in chatroom:
            chatroom['log'] = []
        chatroom['log'].append(message)
        display_messages.refresh(resource)
        chattext.set_value('')
        save_databank()

    async def submit_editdoc(resource):
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        editdocname = databank['players'][idval]['editdocs'][0]
        editdoc = databank['editdocs'][editdocname]
        editdoc['content'] = app.storage.general[editdocname]
        if not 'history' in editdoc:
            editdoc['history'] = []
        editdoc['history'].append({
            'content': editdoc['content'],
            'name': databank['players'][idval]['name'],
            'stamp': time.ctime(),
        })
        save_databank()
        ui.notify('Document submitted')

    tabvars = {}
    await ui.context.client.connected()

    with ui.left_drawer(top_corner=True, bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image(os.path.join(here, 'assets/snowglobe.png')).props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
            with ui.row() as preloginrow:
                login_id = ui.input('ID', placeholder='Player ID').props('size=6')
                ui.button('Connect', on_click=lambda x: set_id(login_id.value))
            with ui.row() as postloginrow:
                postloginrow.set_visibility(False)
                login_numb = ui.label('ID')
                login_name = ui.label('Name')
            ui.input().bind_value(globals(), 'datastep').on_value_change(display_all).set_visibility(False) # Update display on file update
    with ui.header().style('background-color: #B4C7E7'):
        with ui.tabs().classes('w-full') as tabs:
            setup_tabs()
    with ui.tab_panels(tabs).classes('absolute-full'):
        setup_tab_panels()


def snowglobe_interface(host='0.0.0.0', port=8000):
    ui.run(host=host, port=port, title='Snow Globe User Interface',
           favicon=os.path.join(here, 'terminal/favicon.ico'),
           reload=True)


if __name__ in {'__main__', '__mp_main__'}:
    snowglobe_interface()
