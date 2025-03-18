#!/usr/bin/env python3

from nicegui import ui, context, Tailwind

messages = {}

@ui.page('/')
def chat_page():
    # Vertical height adjustment
    # context.client.content.classes('h-[100vh]')
    # ui.add_head_html('<style>.q-textarea.flex-grow .q-field__control { height: 100% }</style>')

    with ui.left_drawer(bordered=True).classes('items-center'):
        with ui.column(align_items='center'):
            ui.image('../terminal/snowglobe.png').props('width=150px').style('border-radius: 5%')
            ui.label('User Interface').style('font-size: 25px; font-weight: bold')
            ui.button('Toggle Full Screen', on_click=ui.fullscreen().toggle)
            ui.button('Toggle Dark Mode', on_click=ui.dark_mode().toggle)
    with ui.tabs().classes('w-full') as tabs:
        chattab = ui.tab('Chat')
        infotab = ui.tab('Info')
    ui.separator()
    with ui.tab_panels(tabs, value=chattab).classes('w-full h-96%'):
        with ui.tab_panel(chattab).classes('h-96%'):
            with ui.splitter(horizontal=True, value="400px", reverse=True).classes('w-full h-96%') as splitter:
                with splitter.before:
                    ui.scroll_area().classes('w-full flex-grow')
                with splitter.after:
                    with ui.column().classes('w-full items-center h-96%'):
                        text = ui.textarea(placeholder='Ask the AI assistant.').classes('w-full border flex-grow').style('padding: 0px 5px')
                        ui.button('Send', on_click=ui.fullscreen().toggle)
                        ui.label('Do not submit sensitive or personal information.').style('font-size: 10px')
        with ui.tab_panel(infotab):
            ui.label('Information')

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(host='0.0.0.0', port=8000, title='Snow Globe Chat', favicon='../terminal/favicon.ico')
