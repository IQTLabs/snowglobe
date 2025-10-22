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

import argparse
import logging
import markdown2
import os
import time
import uuid
import watchfiles

from nicegui import ui, app
from llm_snowglobe.core import Configuration, Database

here = os.path.dirname(os.path.abspath(__file__))
datastep = 0

def main(host="0.0.0.0", port=8000):
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '-c','--config-file', help='location of a yaml config file', action='store',
      default='/config/game.yaml')
    parser.add_argument(
      '-l','--log-file', help='location of a file to log to', action='store',
      default='/home/snowglobe/logs/snowglobe-ui.log')
    args = parser.parse_args()

    cfg_file = args.config_file
    log_file = args.log_file

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.info('Logging started')


    async def detect_updates():
        while True:
            config = Configuration()
            with open(config.game_id_file,'r') as gif:
                ioid = gif.read()

            db = Database(ioid=ioid, path=config.data_dir)
            logger.debug(f'detecting updates to {db.path}')
            if db.path:
                globals()["datastep"] += 1
                async for changes in watchfiles.awatch(db.path):
                    break


    app.on_startup(detect_updates)

    app.add_static_file(url_path="/ai.png", local_file=os.path.join(here, "assets/ai.png"))
    app.add_static_file(
        url_path="/human.png", local_file=os.path.join(here, "assets/human.png")
    )

    def get_database():
        config = Configuration(cfg_file)
        with open(config.game_id_file,'r') as gif:
            ioid = gif.read()

        db = Database(ioid=ioid, path=config.data_dir)
        return db

    @ui.page("/")
    async def ui_page():
        db = get_database()

        async def set_id(idval):
            logger.debug('method set_id entered')
            if len(idval) == 0:
                ui.notify("Enter your ID.")
            elif db.get_name(idval) is None:
                ui.notify("ID not found.")
            else:
                app.storage.tab["id"] = idval
                login_numb.text = app.storage.tab["id"]
                login_name.text = db.get_name(app.storage.tab["id"])
                preloginrow.set_visibility(False)
                postloginrow.set_visibility(True)
                new_resource_check()
                setup_tabs.refresh()
                setup_tab_panels.refresh()
            logger.debug('method set_id exiting')

        def new_resource_check():
            logger.debug('method new_resource_check entered')
            idval = app.storage.tab["id"]
            resource_string = ""
            for resource, resource_type in db.get_assignments(idval):
                resource_string += "|" + resource_type + ":" + resource
            key = "TABSTRING"
            new_resource = key not in tabvars or tabvars[key] != resource_string
            tabvars[key] = resource_string
            logger.debug(f'method new_resource_check returning value {new_resource}')
            return new_resource

        @ui.refreshable
        def setup_tabs():
            logger.debug('method setup_tabs entered')
            if "id" not in app.storage.tab:
                return
            idval = app.storage.tab["id"]
            icon = {
                "chatroom": "chat",
                "weblink": "language",
                "infodoc": "description",
                "notepad": "edit_note",
                "editdoc": "edit",
            }
            default_title = {
                "chatroom": "Chat",
                "weblink": "Data",
                "infodoc": "Info",
                "notepad": "Note",
                "editdoc": "Edit",
            }
            with ui.tabs().classes("w-full") as tabs:
                for resource, resource_type in db.get_assignments(idval):
                    properties = db.get_properties(resource)
                    if "title" in properties:
                        title = properties["title"]
                    else:
                        title = default_title[resource_type]
                    tabvars[resource] = {}
                    tabvars[resource]["tab"] = ui.tab(
                        resource, label=title, icon=icon[resource_type]
                    )
            tabvars["TABCONTAINER"] = tabs
            logger.debug('method setup_tabs exiting')

        @ui.refreshable
        def setup_tab_panels():
            logger.debug('method setup_tab_panels entered')
            if "id" not in app.storage.tab:
                return
            idval = app.storage.tab["id"]
            setup_func = {
                "chatroom": setup_chatroom,
                "weblink": setup_weblink,
                "infodoc": setup_infodoc,
                "notepad": setup_notepad,
                "editdoc": setup_editdoc,
            }
            with ui.tab_panels(tabvars["TABCONTAINER"]).classes("absolute-full") as panels:
                for resource, resource_type in db.get_assignments(idval):
                    setup_func[resource_type](resource)
            if len(tabvars) - 2 == 1:
                panels.set_value(resource)

            logger.debug('method setup_tab_panels exiting')

        def setup_chatroom(resource):
            logger.debug('method setup_chatroom entered')
            with ui.tab_panel(tabvars[resource]["tab"]).classes("h-full"):
                with ui.column().classes("w-full items-center h-full"):
                    tabvars[resource]["message_count"] = 0
                    with ui.scroll_area().classes("w-full h-full border") as tabvars[
                        resource
                    ]["message_window"]:
                        tabvars[resource]["updater"] = ui.refreshable(display_messages)
                        tabvars[resource]["updater"](resource)
                    properties = db.get_properties(resource)
                    placeholder = (
                        properties["instruction"]
                        if "instruction" in properties
                        else "Ask the AI assistant."
                    )
                    tabvars[resource]["chattext"] = (
                        ui.textarea(placeholder=placeholder)
                        .classes("w-full border")
                        .style("height: auto; padding: 0px 5px")
                    )
                    button = ui.button(
                        "Send", on_click=lambda resource=resource: send_message(resource)
                    )
                    if (
                        "readonly" in properties
                        and "|%s|" % db.get_name(app.storage.tab["id"])
                        in properties["readonly"]
                    ):
                        button.enabled = False

            logger.debug('method setup_chatroom exiting')

        def setup_weblink(resource):
            logger.debug('method setup_weblink entered')
            with ui.tab_panel(tabvars[resource]["tab"]).classes("absolute-full"):
                tabvars[resource]["iframe"] = ui.element("iframe").classes(
                    "w-full h-full absolute-full"
                )
                display_weblink(resource)
            logger.debug('method setup_weblink exiting')

        def setup_infodoc(resource):
            logger.debug('method setup_infodoc entered')
            with ui.tab_panel(tabvars[resource]["tab"]).classes("absolute-full"):
                with ui.scroll_area().classes("w-full h-full absolute-full"):
                    tabvars[resource]["updater"] = ui.refreshable(display_infodoc)
                    tabvars[resource]["updater"](resource)
            logger.debug('method setup_infodoc exiting')

        def setup_notepad(resource):
            logger.debug('method setup_notepad entered')
            with ui.tab_panel(tabvars[resource]["tab"]).classes("absolute-full"):
                tabvars[resource]["editor"] = (
                    ui.editor().classes("w-full h-full").props("height=100%")
                )
                tabvars[resource]["editor"]._props.update(
                    toolbar=[
                        [
                            {
                                "label": "Font",
                                "fixedLabel": True,
                                "fixedIcon": True,
                                "list": "no-icons",
                                "options": ["default_font", "arial", "times_new_roman"],
                            },
                            {
                                "label": "Size",
                                "fixedLabel": True,
                                "fixedIcon": True,
                                "list": "no-icons",
                                "options": [
                                    "size-1",
                                    "size-2",
                                    "size-3",
                                    "size-4",
                                    "size-5",
                                    "size-6",
                                    "size-7",
                                ],
                            },
                        ],
                        ["bold", "italic", "underline", "strike", "removeFormat"],
                        ["left", "center", "right", "justify"],
                        ["unordered", "ordered", "link", "hr"],
                        ["undo", "redo"],
                    ]
                )
                tabvars[resource]["editor"]._props.update(
                    fonts={
                        "arial": "Arial",
                        "times_new_roman": "Times New Roman",
                    }
                )
                display_notepad(resource)
            logger.debug('method setup_notepad exiting')

        def setup_editdoc(resource):
            logger.debug('method setup_editdoc entered')
            with ui.tab_panel(tabvars[resource]["tab"]).classes("absolute-full"):
                with ui.column().classes("w-full items-center h-full"):
                    tabvars[resource]["editobj"] = (
                        ui.textarea().classes("w-full").props("input-class=h-96")
                    )
                    display_editdoc(resource)
                    ui.button(
                        "Submit",
                        on_click=lambda resource=resource: submit_editdoc(resource),
                    )
            logger.debug('method setup_editdoc exiting')

        async def display_all():
            logger.debug('method display_all entered')
            if "id" not in app.storage.tab:
                return
            idval = app.storage.tab["id"]
            if new_resource_check():
                setup_tabs.refresh()
                setup_tab_panels.refresh()
                return
            display_func = {
                "chatroom": display_messages,
                "infodoc": display_infodoc,
            }
            for resource, resource_type in db.get_assignments(idval):
                if resource_type in display_func:
                    tabvars[resource]["updater"].refresh(resource)
            logger.debug('method display_all exiting')

        def display_messages(resource):
            logger.debug('method display_messages entered')
            idval = app.storage.tab["id"]
            name = db.get_name(idval)
            chatlog = db.get_chatlog(resource)
            for message in chatlog:
                if "format" not in message or message["format"] == "plaintext":
                    text = message["content"]
                    text_html = False
                elif message["format"] == "markdown":
                    text = markdown2.markdown(message["content"])
                    text_html = True
                elif message["format"] == "html":
                    text = message["content"]
                    text_html = True
                sent = message["name"] == name
                ui.chat_message(
                    text=text,
                    name=message["name"],
                    stamp=message["stamp"],
                    avatar=message["avatar"],
                    sent=sent,
                    text_html=text_html,
                ).classes("w-full")
            if len(chatlog) > tabvars[resource]["message_count"]:
                tabvars[resource]["message_window"].scroll_to(percent=100)
                tabvars[resource]["message_count"] = len(chatlog)

            logger.debug('method display_messages exiting')

        def display_weblink(resource):
            logger.debug('method display_weblink entered')
            properties = db.get_properties(resource)
            if "url" in properties:
                tabvars[resource]["iframe"].props("src=%s" % properties["url"])

            logger.debug('method display_weblink exiting')

        def display_infodoc(resource):
            logger.debug('method display_infodoc entered')
            # idval = app.storage.tab["id"]
            properties = db.get_properties(resource)
            if "content" not in properties:
                return
            if "format" not in properties or properties["format"] == "plaintext":
                ui.label(properties["content"]).style("white-space: pre-wrap").classes(
                    "w-full h-full"
                )
            elif properties["format"] == "markdown":
                ui.markdown(properties["content"]).classes("w-full h-full")
            elif properties["format"] == "html":
                ui.html(properties["content"]).classes("w-full h-full")
            logger.debug('method display_infodoc exiting')

        def display_notepad(resource):
            logger.debug('method display_notepad entered')
            idval = app.storage.tab["id"]
            editor = tabvars[resource]["editor"]
            editor.bind_value(app.storage.general, resource)
            editor.on_value_change(lambda resource=resource: stamp_notepad(resource))
            properties = db.get_properties(resource)
            if (
                "readonly" in properties
                and "|%s|" % db.get_name(idval) in properties["readonly"]
            ):
                editor.enabled = False
            else:
                pass  # ui.timer(300, lambda resource=resource: save_notepad(resource), immediate=False)
            logger.debug('method display_notepad exiting')

        def display_editdoc(resource):
            logger.debug('method display_editdoc entered')
            idval = app.storage.tab["id"]
            editobj = tabvars[resource]["editobj"]
            editobj.bind_value(app.storage.general, resource)
            properties = db.get_properties(resource)
            if (
                "readonly" in properties
                and "|%s|" % db.get_name(idval) in properties["readonly"]
            ):
                editobj.enabled = False

            # Code to reposition cursor, which by default gets moved to
            # the end of the textarea if someone else modifies the text.
            handler_setup = """
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
            """ % tuple(
                [resource] * 3
            )
            text_change_handler = (
                """() => {
                window.editdoccursor.%s.selfChange = true;
            }"""
                % resource
            )
            cursor_move_handler = """() => {
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
            }""" % (
                editobj.id,
                resource,
            )
            ui.run_javascript(handler_setup)
            editobj.on("update:model-value", js_handler=text_change_handler)
            editobj.on("selectionchange", js_handler=cursor_move_handler)
            # editobj.on('update:model-value', lambda: ui.notify('text_change'))
            # editobj.on('selectionchange', lambda: ui.notify('cursor_move'))
            logger.debug('method display_editdoc exiting')

        async def send_message(resource):
            logger.debug('method send_message entered')
            idval = app.storage.tab["id"]
            chattext = tabvars[resource]["chattext"]
            message = {
                "content": chattext.value.strip(),
                "format": "plaintext",
                "name": db.get_name(idval),
                "stamp": time.ctime(),
                "avatar": "human.png",
            }
            db.send_message(resource, **message)
            db.commit()
            tabvars[resource]["updater"].refresh(resource)
            chattext.set_value("")
            logger.debug('method send_message exiting')

        async def stamp_notepad(resource):
            tabvars[resource]["last_modified"] = time.time()

        async def save_notepad(resource):
            logger.debug('method save_notepad entered')
            now = time.time()
            if "last_modified" in tabvars[resource] and (
                "last_saved" not in tabvars[resource]
                or tabvars[resource]["last_saved"] < tabvars[resource]["last_modified"]
            ):
                content = tabvars[resource]["editor"].value
                stamp = time.ctime(tabvars[resource]["last_modified"])
                db.add_property(resource, "content", content)
                db.add_property(resource, "stamp", stamp)
                db.save_text(resource, content, None, stamp)
                db.commit()
                tabvars[resource]["last_saved"] = now
            logger.debug('method save_notepad exiting')

        async def submit_editdoc(resource):
            logger.debug('method submit_editdoc entered')
            idval = app.storage.tab["id"]
            content = app.storage.general[resource]
            name = db.get_name(idval)
            stamp = time.ctime()
            db.add_property(resource, "content", content)
            db.add_property(resource, "stamp", stamp)
            db.save_text(resource, content, name, stamp)
            db.commit()
            ui.notify("Document submitted")
            logger.debug('method submit_editdoc exiting')

        tabvars = {}
        await ui.context.client.connected()

        with ui.left_drawer(top_corner=True, bordered=True).classes("items-center"):
            with ui.column(align_items="center").classes("h-full"):
                ui.image(os.path.join(here, "assets/snowglobe.png")).props(
                    "width=150px"
                ).style("border-radius: 5%")
                ui.label("User Interface").style("font-size: 25px; font-weight: bold")
                ui.chip(
                    "Toggle Full Screen", color="#B4C7E7", on_click=ui.fullscreen().toggle
                )
                ui.chip("Toggle Dark Mode", color="#B4C7E7", on_click=ui.dark_mode().toggle)
                with ui.row() as preloginrow:
                    login_id = ui.input("ID", placeholder="User ID").props("size=5")
                    ui.chip(
                        "Log In", color="#B4C7E7", on_click=lambda: set_id(login_id.value)
                    )
                with ui.row() as postloginrow:
                    postloginrow.set_visibility(False)
                    login_numb = ui.label("ID")
                    login_name = ui.label("Name")
                ui.space().classes("h-100")
                ui.label("Do not input sensitive or personal information.").style(
                    "font-size: 12px; font-style: italic"
                )
                ui.input().bind_value(globals(), "datastep").on_value_change(
                    display_all
                ).set_visibility(
                    False
                )  # Update display on file update
        with ui.header().style("background-color: #B4C7E7"):
            setup_tabs()
        setup_tab_panels()


    def run(host="0.0.0.0", port=8000):
        ui.run(
            host=host,
            port=port,
            title="Snow Globe User Interface",
            favicon=os.path.join(here, "assets/favicon.ico"),
            reload=False,
        )

    run(host, port)

if __name__ in {"__main__", "__mp_main__"}:
    main()
