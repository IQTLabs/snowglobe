#!/usr/bin/env python3

from nicegui import ui

interface_running = False
interface_users = {}

@ui.page('/')
def interface_page():
    ui.context.client.content.classes('h-screen')
    with ui.left_drawer(bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image('terminal/snowglobe.png').props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
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
    ui.run(host=host, port=port, title='Snow Globe User Interface', favicon='terminal/favicon.ico')
    interface_running = True


if __name__ in {'__main__', '__mp_main__'}:
    snowglobe_interface()
