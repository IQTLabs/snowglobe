#!/usr/bin/env python3

import os
import asyncio
from nicegui import ui, app

here = os.path.dirname(os.path.abspath(__file__))

async def load_id(idval):
    app.storage.tab['id'] = idval
    app.storage.tab['logged_in'] = True


@ui.page('/')
async def interface_page():
    await ui.context.client.connected()
    ui.context.client.content.classes('h-screen')
    with ui.left_drawer(bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image(os.path.join(here, 'terminal/snowglobe.png')).props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in', backward=lambda x: not x):
                login_id = ui.number('ID', placeholder='Player ID #', format='%d').props('size=8')
                ui.button('Connect', on_click=lambda x: load_id(login_id.value))
            with ui.row().bind_visibility_from(app.storage.tab, 'logged_in'):
                login_numb = ui.label('ID').bind_text_from(app.storage.tab, 'id')
                login_name = ui.label('Name')
    with ui.tabs().classes('w-full') as tabs:
        chattab = ui.tab('Chat')
        infotab = ui.tab('Info')
    ui.separator()
    with ui.tab_panels(tabs, value=chattab).classes('w-full'):
        with ui.tab_panel(chattab).classes(''):
            with ui.column().classes('w-full items-center'):
                ui.scroll_area().classes('w-full h-[50vh] border')
                ui.textarea(placeholder='Ask the AI assistant.').classes('w-full border').style('height: auto; padding: 0px 5px')
                ui.button('Send', on_click=ui.fullscreen().toggle)
                ui.label('Do not submit sensitive or personal information.').style('font-size: 10px')
        with ui.tab_panel(infotab):
            ui.label('Information')


def snowglobe_interface(host='0.0.0.0', port=8000):
    ui.run(host=host, port=port, title='Snow Globe User Interface',
           favicon=os.path.join(here, 'terminal/favicon.ico'),
           reload=True)


if __name__ in {'__main__', '__mp_main__'}:
    snowglobe_interface()
