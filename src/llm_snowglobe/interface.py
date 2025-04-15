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
app.storage.general['datastep'] = 0

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
        app.storage.general['datastep'] += 1
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
            app.storage.tab['logged_in'] = True
            app.storage.tab['message_count'] = 0
            login_name.text = databank['players'][idval]['name']
            display_all()
            display_editdoc()

    async def send_message():
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        message = {
            'content': chattext.value,
            'format': 'plaintext',
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

    async def submit_editdoc():
        if not 'id' in app.storage.tab:
            return
        idval = app.storage.tab['id']
        editdocname = databank['players'][idval]['editdocs'][0]
        editdoc = databank['editdocs'][editdocname]
        editdoc['content'] = app.storage.general[editdocname]
        if not 'history' in editdoc:
            editdoc['history'] = []
        editdoc['history'].append({
            'text': editdoc['content'],
            'name': databank['players'][idval]['name'],
            'stamp': time.ctime(),
        })
        save_databank()
        ui.notify('Document submitted')

    async def update_editdoc_cursor(event: events.ValueChangeEventArguments):
        # if not 'id' in app.storage.tab:
        #     return
        # print(event)
        # print(editobj.id)
        # await ui.run_javascript('''
        #     const element = getElement(%s).$refs.qRef.getNativeElement();
        #     if (document.activeElement === element) {
        #         const loc = element.selectionStart;
        #         const text = element.value;
        #         const newloc = loc - 1;
        #         element.value = element.value + '()';
        #         element.setSelectionRange(newloc, newloc);
        #         window.loc = newloc;
        #     }
        # ''' % editobj.id)
        #update_editdoc_cursor.post_cursor = editobj.selectionStart
        #print(event.sender.__name__)
        #print(update_editdoc_cursor.post_cursor)
        #editobj.set_selection_range(3,5)
        pass

    def display_all():
        display_messages.refresh()
        display_infodoc.refresh()
        # display_editdoc.refresh() # Do not re-run on databank update

    @ui.refreshable
    def display_messages():
        if 'id' in app.storage.tab:
            idval = app.storage.tab['id']
            name = databank['players'][idval]['name']
            chatroom = databank['chatrooms'][databank['players'][idval]['chatrooms'][0]]
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
                message_window.scroll_to(percent=100)
                app.storage.tab['message_count'] = len(chatroom['log'])

    @ui.refreshable
    def display_infodoc():
        if 'id' in app.storage.tab:
            idval = app.storage.tab['id']
            infodoc = databank['infodocs'][databank['players'][idval]['infodocs'][0]]
            if 'format' not in infodoc or infodoc['format'] == 'plaintext':
                ui.label(infodoc['content']).classes('w-full h-full')
            elif infodoc['format'] == 'markdown':
                ui.markdown(infodoc['content'].replace('\\n', chr(10))).classes('w-full h-full')
            elif infodoc['format'] == 'html':
                ui.html(infodoc['content']).classes('w-full h-full')

    @ui.refreshable
    def display_editdoc():
        if 'id' in app.storage.tab:
            idval = app.storage.tab['id']
            editdocname = databank['players'][idval]['editdocs'][0]
            editobj.bind_value(app.storage.general, editdocname)
            editobj.on_value_change(update_editdoc_cursor)
            if 'readonly' in databank['editdocs'][editdocname]:
                if databank['players'][idval]['name'] in databank['editdocs'][editdocname]['readonly']:
                    editobj.enabled = False

            # ui.run_javascript('''
            # const element = getElement(%s).$refs.qRef.getNativeElement();
            # if (document.activeElement === element) {
            #     const loc = element.selectionStart;
            #     const text = element.value;
            #     const newloc = loc - 1;
            #     element.value = element.value + '()';
            #     element.setSelectionRange(newloc, newloc);
            #     window.loc = newloc;
            # }
            # ''' % editobj.id)

    await ui.context.client.connected()
    app.storage.tab['logged_in'] = False
    ui.add_css('.q-editor__toolbar { display: none }')

    with ui.left_drawer(top_corner=True, bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image(os.path.join(here, 'assets/snowglobe.png')).props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in', backward=lambda x: not x):
                login_id = ui.input('ID', placeholder='Player ID #').props('size=6')
                ui.button('Connect', on_click=lambda x: set_id(login_id.value))
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in'):
                login_numb = ui.label('ID').bind_text_from(app.storage.tab, 'id')
                login_name = ui.label('Name')
            ui.input().bind_value(app.storage.general, 'datastep').on_value_change(display_all).set_visibility(False) # Update display on file update
    with ui.header().style('background-color: #B4C7E7'):
        with ui.tabs().classes('w-full') as tabs:
            chattab = ui.tab('Chat')
            infotab = ui.tab('Info')
            edittab = ui.tab('Edit')
    with ui.tab_panels(tabs, value=chattab).classes('absolute-full'):
        with ui.tab_panel(chattab).classes('h-full'):
            with ui.column().classes('w-full items-center h-full'):
                with ui.scroll_area().classes('w-full h-full border') as message_window:
                    display_messages()
                chattext = ui.textarea(placeholder='Ask the AI assistant.').classes('w-full border').style('height: auto; padding: 0px 5px')
                ui.button('Send', on_click=send_message)
                ui.label('Do not send sensitive or personal information.').style('font-size: 10px')
        with ui.tab_panel(infotab).classes('absolute-full'):
            with ui.scroll_area().classes('w-full h-full absolute-full'):
                display_infodoc()
        with ui.tab_panel(edittab).classes('absolute-full'):
            with ui.column().classes('w-full items-center h-full'):
                #editobj = ui.editor().classes('w-full h-full')
                editobj = ui.textarea().classes('w-full').props('input-class=h-80')
                #editobj = ui.element('textarea').classes('w-full h-full').style('border: 1px solid #e5e7eb; padding: 5px')
                #editobj = ui.input().classes('w-full h-full')
                display_editdoc()
                ui.button('Submit', on_click=submit_editdoc)
                ui.label('Do not input sensitive or personal information.').style('font-size: 10px')


def snowglobe_interface(host='0.0.0.0', port=8000):
    ui.run(host=host, port=port, title='Snow Globe User Interface',
           favicon=os.path.join(here, 'terminal/favicon.ico'),
           reload=True)


if __name__ in {'__main__', '__mp_main__'}:
    snowglobe_interface()
