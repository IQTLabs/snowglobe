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

import asyncio
import langchain_core.documents
import langchain_core.tools


def AskTool(name, desc, player, history=None, level=0):
    def ask(query: str) -> str:
        return asyncio.run(
            player.return_output(
                name=player.name,
                persona=player.persona,
                history=history,
                query=query,
                level=level + 1,
            )
        )

    async def aask(query: str) -> str:
        return await player.return_output(
            name=player.name,
            persona=player.persona,
            history=history,
            query=query,
            level=level + 1,
        )

    tool = langchain_core.tools.StructuredTool.from_function(
        func=ask,
        coroutine=aask,
        name=name,
        description=desc,
        return_direct=False,
        metadata={"level": level},
    )
    return tool
