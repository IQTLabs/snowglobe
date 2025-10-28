#!/usr/bin/env python3

#   Copyright 2023-2025 IQT Labs LLC
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
import sqlite3
import watchfiles

from .history import History

class Database:

    @staticmethod
    def get_db_by_ioid(ioid, path=None):
        db = None
        db_file_path = os.path.join(path, f"snowglobe_{ioid}.db")
        if os.path.exists(db_file_path) and os.path.isfile(db_file_path):
            db = Database(ioid, path)

        return db

    def __init__(self, ioid, path=None, initialize=False):
        # breakpoint()
        db_file_path = os.path.join(path, f"snowglobe_{ioid}.db")
        if (os.path.exists(db_file_path) and os.path.isfile(db_file_path)) or initialize:
            self.path = db_file_path
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        # if not self.con:
        #     breakpoint()
        self.cur = self.con.cursor()

        if initialize:
            self.create()

    def create(self):
        self.cur.execute("create table if not exists players(id primary key, name)")
        self.cur.execute(
            "create table if not exists resources(resource primary key, type)"
        )
        self.cur.execute(
            "create table if not exists assignments(ord integer primary key, id, resource)"
        )
        self.cur.execute(
            "create table if not exists properties(resource, property, value, primary key (resource, property))"
        )
        self.cur.execute(
            "create table if not exists chatlog(ord integer primary key, resource, content, format, name, stamp, avatar)"
        )
        self.cur.execute(
            "create table if not exists textlog(ord integer primary key, resource, content, name, stamp)"
        )
        self.cur.execute(
            "create table if not exists histlog(ord integer primary key, resource, content, name)"
        )

    def add_player(self, pid, name):
        self.cur.execute("replace into players values(?, ?)", (pid, name))

    def add_resource(self, resource, rtype):
        self.cur.execute("replace into resources values(?, ?)", (resource, rtype))

    def assign(self, pid, resource):
        self.cur.execute(
            "delete from assignments where id = ? and resource = ?", (pid, resource)
        )
        self.cur.execute("replace into assignments values(null, ?, ?)", (pid, resource))

    def add_property(self, resource, rproperty, value):
        self.cur.execute(
            "replace into properties values(?, ?, ?)", (resource, rproperty, value)
        )

    def get_name(self, pid):
        res = self.cur.execute(
            "select name from players where id == ?", (pid,)
        ).fetchone()
        return res[0] if res is not None else None

    def get_assignments(self, pid):
        res = self.cur.execute(
            "select assignments.resource, type from resources join assignments on resources.resource = assignments.resource where id == ? order by ord",
            (pid,),
        ).fetchall()
        return res

    def get_properties(self, resource):
        res = self.cur.execute(
            "select property, value from properties where resource == ?", (resource,)
        ).fetchall()
        return dict(res)

    def default_chatroom(self, ioid):
        # Return player's default chatroom; create one if needed
        for resource, resource_type in self.get_assignments(ioid):
            if resource_type == "chatroom":
                return resource
        chatroom = "default_%s" % ioid
        self.add_resource(chatroom, "chatroom")
        self.assign(ioid, chatroom)
        self.commit()
        return chatroom

    def get_chatlog(self, chatroom=None):
        if chatroom is not None:
            res = self.cur.execute(
                "select content, format, name, stamp, avatar from chatlog where resource == ? order by ord",
                (chatroom,),
            ).fetchall()
            return [
                dict(zip(("content", "format", "name", "stamp", "avatar"), x))
                for x in res
            ]
        else:
            res = self.cur.execute(
                "select resource as chatroom, content, format, name, stamp, avatar from chatlog order by ord"
            ).fetchall()
            return [
                dict(
                    zip(("chatroom", "content", "format", "name", "stamp", "avatar"), x)
                )
                for x in res
            ]

    def get_history(self, resource, target=None):
        if target is not None:
            history = target
            history.clear()
        else:
            history = History()
        res = self.cur.execute(
            "select content, name from histlog where resource == ? order by ord",
            (resource,),
        ).fetchall()
        for x in res:
            history.add(x[1], x[0])
        if target is not None:
            return
        else:
            return history

    def send_message(self, chatroom, content, format, name, stamp, avatar):
        self.cur.execute(
            "insert into chatlog values(null, ?, ?, ?, ?, ?, ?)",
            (chatroom, content, format, name, stamp, avatar),
        )

    def save_text(self, resource, content, name, stamp):
        self.cur.execute(
            "insert into textlog values(null, ?, ?, ?, ?)",
            (resource, content, name, stamp),
        )

    def set_history(self, resource, history):
        self.cur.execute("delete from histlog where resource = ?", (resource,))
        for entry in history.entries:
            self.cur.execute(
                "insert into histlog values(null, ?, ?, ?)",
                (resource, entry["text"], entry["name"]),
            )

    def commit(self):
        self.con.commit()

    async def wait(self):
        async for changes in watchfiles.awatch(self.path):
            break

    def __del__(self):
        if hasattr(self, 'con'):
            self.con.close()
