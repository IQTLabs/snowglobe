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
import re
import sys
import yaml
import json
import time
import uuid
import random
import shutil
import asyncio
import inspect
import sqlite3
import readline
import watchfiles
import numpy as np
import platformdirs
import transformers
import urllib.parse
import urllib.request
import langchain_openai
import langchain_chroma
import langchain.prompts
import langchain.storage
import langchain.retrievers
import langchain_huggingface
import langchain_text_splitters
import langchain_core.documents
import langchain_community.llms
import langchain_community.embeddings

verbose = 2


def settings():
    s = {}
    # Standard paths
    s['config_dir'] = platformdirs.user_config_dir('snowglobe')
    s['cache_dir'] = platformdirs.user_cache_dir('snowglobe')
    s['data_dir'] = platformdirs.user_data_dir('snowglobe')
    s['menu_file'] = 'llms.yaml'
    s['menu_path'] = os.path.join(s['config_dir'], s['menu_file'])

    # Default model
    s['default_source'] = 'llamacpp'
    s['default_model'] = 'mistral-7b-openorca'
    s['default_url'] = 'https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_M.gguf'
    s['default_file'] = os.path.basename(
        urllib.parse.urlparse(s['default_url']).path)
    s['default_path'] = os.path.join(s['cache_dir'], s['default_file'])
    return s


def read_yaml(path):
    # Reads YAML file, expressing empty or nonexistent file as empty dictionary
    if os.path.exists(path):
        with open(path, 'r') as obj:
            content = yaml.safe_load(obj)
        if content is None:
            content = {}
    else:
        content = {}
    return content


def write_yaml(data, path):
    # Writes YAML file, creating empty directory if needed
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as obj:
        yaml.dump(data, obj, default_flow_style=False, sort_keys=False)


def config(menu=None, source=None, model=None, url=None, path=None,
           update_menu=True, download_weights=True):
    s = settings()
    menu = menu if menu is not None else s['menu_path']
    source = source if source is not None else s['default_source']
    model = model if model is not None else s['default_model']
    url = url if url is not None else s['default_url']
    path = path if path is not None else s['default_path']

    # Update menu of source+model options
    if update_menu:
        options = read_yaml(menu)
        if source not in options:
            options[source] = {}
        options[source][model] = path
        write_yaml(options, menu)

    # Download model weights
    if download_weights:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print('Downloading model weights... ', end='', flush=True)
        urllib.request.urlretrieve(url, path)
        print('Done', flush=True)


class Database():
    def __init__(self, path=None):
        self.path = path if path is not None else 'snowglobe.db'
        self.con = sqlite3.connect(self.path)
        self.cur = self.con.cursor()
        self.create()
    def create(self):
        self.cur.execute("create table if not exists players(id primary key, name)")
        self.cur.execute("create table if not exists resources(resource primary key, type)")
        self.cur.execute("create table if not exists assignments(ord integer primary key, id, resource)")
        self.cur.execute("create table if not exists properties(resource, property, value, primary key (resource, property))")
        self.cur.execute("create table if not exists chatlog(ord integer primary key, resource, content, format, name, stamp, avatar)")
        self.cur.execute("create table if not exists textlog(ord integer primary key, resource, content, name, stamp)")
        self.cur.execute("create table if not exists histlog(ord integer primary key, resource, content, name)")
    def add_player(self, pid, name):
        self.cur.execute("replace into players values(?, ?)", (pid, name))
    def add_resource(self, resource, rtype):
        self.cur.execute("replace into resources values(?, ?)",
                         (resource, rtype))
    def assign(self, pid, resource):
        self.cur.execute("delete from assignments where id = ? and resource = ?", (pid, resource))
        self.cur.execute("replace into assignments values(null, ?, ?)",
                         (pid, resource))
    def add_property(self, resource, rproperty, value):
        self.cur.execute("replace into properties values(?, ?, ?)",
                         (resource, rproperty, value))
    def get_name(self, pid):
        res = self.cur.execute("select name from players where id == ?",
                               (pid,)).fetchone()
        return res[0] if res is not None else None
    def get_assignments(self, pid):
        res = self.cur.execute("select assignments.resource, type from resources join assignments on resources.resource = assignments.resource where id == ? order by ord", (pid,)).fetchall()
        return res
    def get_properties(self, resource):
        res = self.cur.execute("select property, value from properties where resource == ?", (resource,)).fetchall()
        return dict(res)
    def get_chatlog(self, chatroom=None):
        if chatroom is not None:
            res = self.cur.execute("select content, format, name, stamp, avatar from chatlog where resource == ? order by ord", (chatroom,)).fetchall()
            return [dict(zip(('content', 'format', 'name', 'stamp', 'avatar'), x)) for x in res]
        else:
            res = self.cur.execute("select resource as chatroom, content, format, name, stamp, avatar from chatlog order by ord").fetchall()
            return [dict(zip(('chatroom', 'content', 'format', 'name', 'stamp', 'avatar'), x)) for x in res]
    def send_message(self, chatroom, content, format, name, stamp, avatar):
        self.cur.execute("insert into chatlog values(null, ?, ?, ?, ?, ?, ?)", (chatroom, content, format, name, stamp, avatar))
    def save_text(self, resource, content, name, stamp):
        self.cur.execute("insert into textlog values(null, ?, ?, ?, ?)", (resource, content, name, stamp))
    def commit(self):
        self.con.commit()
    async def wait(self):
        async for changes in watchfiles.awatch(self.path):
            break
    def __del__(self):
        self.con.close()

db = Database()


class LLM():
    def __init__(
            self, source=None, model=None, menu=None, gen=None, embed=None):

        s = settings()
        self.menu = menu if menu is not None else s['menu_path']
        self.source = source if source is not None else s['default_source']
        self.model = model if model is not None else s['default_model']
        self.gen = gen if gen is not None else True
        self.embed = embed if embed is not None else False

        if not self.source in ['openai', 'azure']:
            options = read_yaml(self.menu)
            # When using the default model and standard menu path,
            # if the model is not found then auto-install it.
            if (self.source not in options or self.model not in options[self.source]) and self.source == s['default_source'] and self.model == s['default_model'] and self.menu == s['menu_path']:
                config()
                options = read_yaml(self.menu)
            self.model_path = options[self.source][self.model]
            self.model_path = os.path.expanduser(self.model_path)
            if not os.path.isabs(self.model_path):
                self.model_path = os.path.join(self.menu, self.model_path)
        elif self.source in ['azure']:
            options = read_yaml(self.menu)
            self.azure_deployment = options[self.source][self.model][
                'deployment']
            self.azure_endpoint = options[self.source][self.model]['endpoint']
            self.azure_version = options[self.source][self.model]['version']

        if self.source == 'openai':

            # Model Source: OpenAI (Cloud)
            if self.gen:
                self.llm = langchain_openai.ChatOpenAI(
                    model_name=self.model,
                    streaming=True,
                )
            if self.embed:
                self.embeddings = langchain_openai.OpenAIEmbeddings()

        elif self.source == 'azure':

            # Model Source: Azure OpenAI (Cloud)
            if self.gen:
                self.llm = langchain_openai.AzureChatOpenAI(
                    azure_deployment=self.azure_deployment,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_version,
                    streaming=True,
                )
            if self.embed:
                self.embeddings = langchain_openai.AzureOpenAIEmbeddings(
                    azure_deployment=self.azure_deployment,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_version,
                )

        elif self.source == 'llamacpp':

            # Model Source: llama.cpp (Local)
            if self.gen:
                self.llm = langchain_community.llms.LlamaCpp(
                    model_path=self.model_path,
                    n_gpu_layers=-1,
                    seed=random.randint(0, sys.maxsize),
                    n_ctx=32768,
                    n_batch=512,
                    f16_kv=True,
                    max_tokens=1000,
                    verbose=False,
                )
            if self.embed:
                self.embeddings = \
                    langchain_community.embeddings.LlamaCppEmbeddings(
                        model_path=self.model_path, n_gpu_layers=-1,
                        n_batch=512, n_ctx=8192, f16_kv=True, verbose=False)
            self.serial = True

        elif self.source == 'huggingface':

            # Model Source: Hugging Face (Local)
            if self.gen:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_path, device_map='auto')
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.model_path, device_map='auto')
                tokenizer.pad_token = tokenizer.eos_token
                streamer = transformers.TextStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True) \
                    if verbose >= 1 else None
                pipeline = transformers.pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device_map='auto',
                    max_new_tokens=2048,
                    repetition_penalty=1.05,
                    return_full_text=False,
                    streamer=streamer,
                    do_sample=True,
                )
                self.llm = langchain_huggingface.llms.HuggingFacePipeline(
                    pipeline=pipeline)
            if self.embed:
                self.embeddings = \
                    langchain_huggingface.embeddings.HuggingFaceEmbeddings(
                        model_name=self.model_path, show_progress=True)

        self.bound = {'stop': '##'}


async def gather_plus(*args):
    items = np.array(args)
    flags = np.array([inspect.isawaitable(x) for x in items])
    if sum(flags) == 0:
        return items
    awaitables = items[flags]
    outputs = await asyncio.gather(*awaitables)
    items[flags] = outputs
    return items


class History():
    def __init__(self):
        self.entries = []
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        history_part = History()
        if isinstance(index, slice):
            history_part.entries = self.entries[index]
        elif isinstance(index, int):
            history_part.entries = [self.entries[index]]
        return history_part
    def add(self, name, text):
        self.entries.append({'name': name, 'text': text})
    async def concrete(self):
        texts = [entry['text'] for entry in self.entries]
        texts = await gather_plus(*texts)
        for i in range(len(texts)):
            self.entries[i]['text'] = texts[i]
    async def str(self, name=None):
        await self.concrete()
        return '\n\n'.join([
            ('You' if entry['name'] == name else entry['name'])
            + ':\n\n' + entry['text'] for entry in self.entries])
    async def textonly(self):
        await self.concrete()
        return '\n\n'.join([entry['text'] for entry in self.entries])
    def clear(self):
        self.entries = []
    def copy(self):
        history_copy = History()
        history_copy.entries = self.entries.copy()
        return history_copy


class EphemeralDir():
    def __init__(self, path, mode=0o777, exist_ok=False):
        self.path = path
        os.makedirs(path, mode=mode, exist_ok=exist_ok)
    def __del__(self):
        shutil.rmtree(self.path) # Note: Parent dirs of path not deleted


class ClassicRAG():
    def rag_init(self, loader, chunk_size=None, chunk_overlap=None,
                 count=None):
        if chunk_size is None:
            chunk_size = 1000
        if chunk_overlap is None:
            chunk_overlap = 200
        if count is None:
            count = 1

        docs = loader.load()
        splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = splitter.split_documents(docs)
        collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', self.name)[:63]
        vectorstore = langchain_chroma.Chroma.from_documents(
            documents=splits, embedding=self.rag_llm.embeddings,
            collection_name=collection_name)
        self.retriever = vectorstore.as_retriever(search_kwargs={'k': count})

    def rag_invoke(self, text):
        return self.retriever.invoke(text)

class DescriptionRAG():
    def rag_init(self, loader, chunk_size=None, chunk_overlap=None,
                 count=None):
        if chunk_size is None:
            chunk_size = 1000
        if chunk_overlap is None:
            chunk_overlap = 200
        if count is None:
            count = 1

        # Create splits
        docs = loader.load()
        splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = splitter.split_documents(docs)

        # Create retriever
        collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', self.name)[:63]
        vectorstore = langchain_chroma.Chroma(
            embedding_function=self.rag_llm.embeddings,
            collection_name=collection_name)
        bytestore = langchain.storage.InMemoryByteStore()
        id_key = 'split'
        retriever = langchain.retrievers.multi_vector.MultiVectorRetriever(
            vectorstore=vectorstore, byte_store=bytestore, id_key=id_key,
            search_kwargs={'k': count})

        # Create descriptions
        split_ids = [str(uuid.uuid4()) for _ in splits]
        description_strings = []
        for split in splits:
            output = self.rag_description(split.page_content)
            description_strings.append(output)
        description_docs = [langchain_core.documents.Document(
            page_content=description_string, metadata={id_key: split_ids[i]})
            for i, description_string in enumerate(description_strings)]

        # Store splits and descriptions to retriever
        retriever.vectorstore.add_documents(description_docs)
        retriever.docstore.mset(list(zip(split_ids, splits)))
        self.rag_retriever = retriever

    def rag_description(self, text):
        #template = 'Question:\n\nGive some keywords to describe the situation or problem faced by {name} in the following text.  The keywords should be in an unnumbered, comma-separated list.\n\n{text}\n\nAnswer:\n\nKeywords:'
        template = 'Question:\n\nIn one sentence, describe the problem {name} is facing in the following text.\n\n{text}\n\nAnswer:\n\n'
        variables = {'name': self.name, 'text': text}
        bind = {'stop': ['\n\n']}
        output = self.return_output(
            bind=bind,
            template=template, variables=variables
        )
        return output
        
    def rag_invoke(self, text):
        description = self.rag_description(text)
        splits = self.rag_retriever.invoke(description)
        return splits

class RAG(ClassicRAG):
    pass


class Intelligent():
    def setup_kind(self, kind):
        if kind == 'ai':
            pass
        elif kind == 'human':
            self.interface_setup()
        elif kind == 'api':
            self.set_api_id()
        elif kind == 'preset':
            if self.presets is not None:
                self.preset_generator = self.set_preset_generator(self.presets)

    async def return_output(self, kind=None, bind=None,
                            template=None, variables=None, **kwargs):
        # Set defaults
        if kind is None:
            kind = self.kind

        # Use intelligent entity (AI or human) to generate output
        if template is None and variables is None:
            template, variables = await self.return_template(**kwargs)
        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        if kind == 'ai':
            output = await self.return_from_ai(prompt, variables, bind=bind)
        elif kind == 'human':
            output = await self.return_from_human(prompt, variables)
        elif kind == 'api':
            output = await self.return_from_api(prompt, variables)
        elif kind == 'preset':
            output = await self.return_from_preset()
        return output

    async def return_template(
            self, name=None,
            persona=None, reminder=None,
            rag=None, rag_intro=None, rag_query=None,
            history=None, history_over=None, history_merged=None,
            responses=None, responses_intro=None,
            query=None, query_format=None, query_subtitle=None
    ):
        # Punctuation edit
        if persona is not None:
            if persona[-1:] == '.':
                persona = persona[:-1]

        # Create template and variables
        template = ''
        variables = {}
        if persona is not None:
            template += '### You are {persona}.\n\n'
            variables['persona'] = persona
        if rag is not None and history is not None:
            if rag_intro is None:
                rag_intro = 'Here is background information to inform your response'
            if rag_query is None:
                rag_query = history.entries[-1]['text']
            docs = self.rag_invoke(rag_query)
            ragstring = '\n'.join(doc.page_content for doc in docs)
            template += '### ' + rag_intro + ':\n\n{rag}\n\n'
            variables['rag'] = ragstring
        if history is not None:
            if not history_over:
                history_intro = 'This is what has happened so far'
            else:
                history_intro = 'This is what happened'
            template += '### ' + history_intro + ':\n\n{history}\n\n'
            if isinstance(history, str):
                variables['history'] = history
            elif not history_merged:
                variables['history'] = await history.str(name=name)
            else:
                variables['history'] = await history.textonly()
        if responses is not None:
            template += '### ' + responses_intro + ':\n\n{responses}\n\n'
            variables['responses'] = await responses.str(name=name)
        if query_format is None or query_format == 'twoline':
            template += '### Question:\n\n{query}'
            if reminder == 2 and persona is not None:
                template += ' (Remember, you are {persona}.)'
            elif reminder == 1 and name is not None:
                template += ' (Remember, you are {name}.)'
                variables['name'] = name
            template += '\n\n### Answer:\n\n'
        elif query_format == 'oneline':
            template += '### {query}:\n\n'
        elif query_format == 'twoline_simple':
            template += 'Question: {query}\n\nAnswer: '
        elif query_format == 'oneline_simple':
            template += '{query}'
        variables['query'] = query
        if query_subtitle is not None:
            template += '{subtitle}:\n\n'
            variables['subtitle'] = query_subtitle
        return template, variables

    async def return_from_ai(self, prompt, variables, max_tries=64, bind=None):
        llm = self.llm.llm.bind(**self.llm.bound)
        if bind is not None:
            llm = llm.bind(**bind)
        chain = prompt | llm
        if verbose >= 3:
            print('v' * 80)
            print(prompt.format(**variables))
            print('^' * 80)
        for i in range(max_tries):
            if not verbose >= 1 or self.llm.source == 'huggingface':
                if hasattr(self.llm, 'serial') and self.llm.serial == True:
                    output = chain.invoke(variables)
                else:
                    output = await chain.ainvoke(variables)
                if self.llm.source in ['openai', 'azure']:
                    output = output.content
                output = output.strip()
            else:
                def handle(chunk):
                    if self.llm.source in ['openai', 'azure']:
                        chunk = chunk.content
                    print(chunk, end='', flush=True)
                    return chunk
                output = ''
                if hasattr(self.llm, 'serial') and self.llm.serial == True:
                    for chunk in chain.stream(variables):
                        output += handle(chunk)
                else:
                    async for chunk in chain.astream(variables):
                        output += handle(chunk)
                output = output.strip()
            if len(output) > 0:
                break
        if verbose >= 1:
            print()
        return output

    async def return_from_human(self, prompt, variables):
        # Determine chatroom to use.  If user has more than one, use the first.
        # If user has none, create one.
        chatroom = None
        for resource, resource_type in db.get_assignments(self.interface_label):
            if resource_type == 'chatroom':
                chatroom = resource
                break
        if chatroom is None:
            chatroom = '%s_%s_default' % (self.name, self.interface_label)
            db.add_resource(chatroom, 'chatroom')
            db.assign(self.interface_label, chatroom)
            db.commit()

        # Send prompt.  Create a temporary control just to send the message.
        sender = Control(llm=None)
        content = prompt.format(**variables)
        sender.interface_send_message(chatroom, content)

        # Get response
        answer = await self.interface_get_message(chatroom)
        return answer

    async def return_from_api(self, prompt, variables, delay=2):
        prompt_path = self.get_api_path(False)
        answer_path = self.get_api_path(True)
        self.api_count += 1

        # Write prompt to disk
        prompt_content = prompt.format(**variables)
        prompt_json = {'content': prompt_content}
        if not os.path.exists(os.path.dirname(prompt_path)):
            os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, 'w') as f:
            json.dump(prompt_json, f)

        # Read answer from disk
        while not os.path.exists(answer_path):
            await asyncio.sleep(delay)
            if verbose >= 2:
                print('Awaiting %s [ID %i # %i]'
                      % (self.name, self.api_label, self.api_count - 1))
        with open(answer_path, 'r') as f:
            answer_json = json.load(f)
        answer_content = answer_json['content']
        return answer_content

    async def return_from_preset(self):
        return next(self.preset_generator, '')

    def set_preset_generator(self, presets):
        for preset in presets:
            if verbose >= 2:
                print(preset)
            yield preset

    def set_api_id(self):
        self.api_label = random.randint(100000, 999999)
        self.api_count = 0
        if verbose >= 2:
            print('ID %i : %s' % (self.api_label, self.name))
        intro_path = self.get_api_path(False)
        intro_json = {'name': self.name}
        if not os.path.exists(os.path.dirname(intro_path)):
            self.api_log = EphemeralDir(
                os.path.dirname(intro_path), exist_ok=True)
        with open(intro_path, 'w') as f:
            json.dump(intro_json, f)
        self.api_count += 1

    def get_api_path(self, answer=False, base_path=None):
        if base_path is None:
            base_path = settings()['data_dir']
        return os.path.join(base_path, str(self.api_label), '%i_%i_%s.json'
                            % (self.api_label, self.api_count,
                               'answer' if answer else 'prompt'))

    def interface_setup(self):
        # Assign ID
        if self.ioid is not None:
            self.interface_label = str(self.ioid)
        else:
            self.interface_label = str(random.randint(100000, 999999))
        if verbose >= 2:
            print('ID %s : %s' % (self.interface_label, self.name))

        # Export info for UI interface
        db.add_player(self.interface_label, self.name)
        if self.iodict is not None:
            for resource_type in ['chatrooms', 'weblinks', 'infodocs',
                                  'notepads', 'editdocs']:
                if resource_type in self.iodict:
                    for resource in self.iodict[resource_type]:
                        db.add_resource(resource, resource_type[:-1])
                        db.assign(self.interface_label, resource)
        db.commit()

    def interface_send_message(self, chatroom, content, fmt=None, cc=None):
        if fmt is None:
            fmt = 'markdown'
        avatar = 'ai.png' if self.kind == 'ai' else 'human.png'
        message = {'content': content,
                   'format': fmt,
                   'name': self.name,
                   'stamp': time.ctime(),
                   'avatar': avatar}
        if cc is None:
            destinations = [chatroom]
        elif isinstance(cc, str):
            destinations = [chatroom, cc]
        else:
            destinations = [chatroom].append(cc)
        for destination in destinations:
            db.send_message(destination, **message)
        db.commit()

    async def interface_get_message(self, chatroom):
        while True:
            log = db.get_chatlog(chatroom)
            if len(log) > 0 and log[-1]['name'] == self.name:
                answer = log[-1]['content']
                return answer
            await db.wait()

    async def multiple_choice(self, query, answer, mc):
        mc_parts = ["'" + x + "'," for x in mc]
        if len(mc_parts) == 2:
            mc_parts[0] = mc_parts[0][:-1]
        mc_parts[-1] = 'or ' + mc_parts[-1][:-1]
        mc_string = ' '.join(mc_parts)
        template = 'Question: {query}\n\nAnswer: {answer}\n\nQuestion: Which multiple choice response best summarizes the previous answer?: {mc_string}.\n\nAnswer: '
        variables = {'query': query, 'answer': answer, 'mc_string': mc_string}
        bind = {'stop': ['\n\n']}
        output = await self.return_output(
            bind=bind,
            template=template, variables=variables
        )
        output = output.strip().strip('"' + "'" + '.,;').lower()
        mc_ref = [x.lower() for x in mc]
        if output in mc_ref:
            output = mc[mc_ref.index(output)]
        else:
            search_flags = [x in output for x in mc_ref]
            if sum(search_flags) == 1:
                output = mc[search_flags.index(True)]
            else:
                output = ''
        return output

    async def chat_backend(self, name='Assistant', persona=None, rag=None,
                           history=None):
        chatlog = History()
        nb = 2
        username = 'User'
        chat_intro = 'This is a conversation about what happened'
        if verbose >= 2:
            instructions = 'Chat | Press Enter twice after each message or to exit.'
            print()
            print('-' * len(instructions))
            print(instructions)
            print('-' * len(instructions))
        while True:
            # Get user input, which may be multiline if no line is blank
            usertext = ''
            while True:
                userline = input()
                usertext += userline + '\n'
                if len(usertext) >= nb and usertext[-nb:] == '\n' * nb:
                    break
            if len(usertext) == nb and usertext[-nb:] == '\n' * nb:
                break
            usertext = usertext.strip()
            chatlog.add(username, usertext)

            # Get response
            bind = {'stop': [username + ':', name + ':', 'Narrator:']}
            output = await self.return_output(
                bind=bind,
                persona=persona,
                rag=rag,
                history=history, history_over=True,
                responses=chatlog, responses_intro=chat_intro,
                query=name + ':\n\n', query_format='oneline_simple'
            )
            print()
            chatlog.add(name, output)

    async def chat_response(self, chatlog, name='Assistant', persona=None,
                            rag=None, history=None, participants=None):
        # Get single response, given prexisting chatlog
        chat_intro = 'This is a conversation about what happened'
        if participants is None:
            participants = set([entry['name'] for entry in chatlog.entries])
            participants.add(name)
            participants.add('Narrator')
        bind = {'stop': [p + ':' for p in participants]}
        output = await self.return_output(
            bind=bind,
            persona=persona,
            rag=rag,
            history=history, history_over=True,
            responses=chatlog, responses_intro=chat_intro,
            query=name + ':\n\n', query_format='oneline_simple'
        )
        return output

    async def join_chatroom(self, chatroom, rag=None, history=None):
        while True:
            log = db.get_chatlog(chatroom)
            # Respond unless most recent message was from self
            if len(log) > 0 and log[-1]['name'] != self.name:
                # Construct message log
                chatlog = History()
                for logitem in log:
                    chatlog.add(logitem['name'], logitem['content'])
                # Respond
                output = await self.chat_response(
                    chatlog, name=self.name,
                    persona=self.persona,
                    rag=None, history=None)
                self.interface_send_message(chatroom, output)
            await db.wait()


class Stateful():
    def record_narration(self, narration, timestep=None, index=None):
        if timestep is None and index is None:
            label = 'Narrator'
        else:
            part1 = timestep.title() + ' ' if timestep is not None else ''
            part2 = str(index) if index is not None else str(len(self.history))
            label = part1 + part2
        self.history.add(label, narration)

    def record_response(self, player_name, player_response):
        self.history.add(player_name, player_response)


class Control(Intelligent, Stateful, RAG):
    def __init__(
            self, source=None, model=None, menu=None, gen=None, embed=None,
            loader=None, chunk_size=None, chunk_overlap=None, count=None,
            llm=None, rag_llm=None, ioid=None, iodict=None, presets=None,
    ):
        self.llm = LLM(
            source=source, model=model, menu=menu, gen=gen, embed=embed
        ) if llm is None else llm
        self.name = 'Control'
        self.kind = 'ai'
        self.persona = None
        self.ioid = ioid
        self.iodict = iodict
        self.presets = presets
        self.history = History()
        self.setup_kind(self.kind)

        if loader is not None:
            self.rag_llm = rag_llm if rag_llm is not None else self.llm
            self.rag_init(loader, chunk_size, chunk_overlap, count)
            self.rag = True
        else:
            self.rag = None

    async def __call__(self):
        raise Exception('! Override this method in the subclass for your specific scenario.')

    def run(self):
        asyncio.run(self())

    def header(self, title, h=0, width=80):
        print()
        if h == 0:
            print('+-' + '-' * min(len(title), width - 4) + '-+')
            print('| ' + title + ' |')
            print('+-' + '-' * min(len(title), width - 4) + '-+')
        elif h == 1:
            print('-' * min(len(title), width))
            print(title)
            print('-' * min(len(title), width))
        else:
            print(title)

    async def adjudicate(self, history=None, responses=None, query=None,
                         nature=True, timestep='week', mode=[]):
        responses_intro = 'These are the plans for each person or group'
        if 'geopol' in mode:
            responses_intro = 'These are the plans ordered by each leader'
        if query is None:
            query = 'Weave these plans into a cohesive narrative of what happens in the next ' + timestep + '.'
            if 'geopol' in mode:
                query = 'Describe these plans being carried out, assuming the leaders above issue no further orders.'
            if random.random() < nature:
                query += ' Include unexpected consequences.'
        output = await self.return_output(
            history=history,
            responses=responses, responses_intro=responses_intro,
            query=query, query_format='oneline'
        )
        if 'summarize' in mode:
            print('\n### Summary\n')
            template = 'Give a short summary of the News.\n\n### History:\n\n{history}\n\n### News:\n\n{news}\n\n### Summary of the News:\n\n'
            variables = {'history': await history.textonly(), 'news': output}
            output = self.return_output(
                template=template, variables=variables
            )
        return output

    async def assess(self, history=None, responses=None, query=None,
                     mc=None, short=False):
        responses_intro = 'Questions about what happened'
        if responses is None:
            query_format = 'twoline'
        else:
            query_format = 'twoline_simple'
        bind = {'stop': ['\n\n']} if short else None
        output = await self.return_output(
            bind=bind,
            history=history, history_over=True,
            responses=responses, responses_intro=responses_intro,
            query=query, query_format=query_format
        )
        if mc is not None:
            output = await self.multiple_choice(query, output, mc)
        return output

    def chat(self, history=None):
        name = self.name
        persona = 'the Control (a.k.a. moderator) of a simulated scenario'
        return self.chat_backend(name=name, persona=persona, history=history)

    async def create_scenario(self, query=None, clip=0):
        if query is None:
            raise Exception('Query required to create scenario.')
        output = await self.return_output(
            query=query, query_format='twoline_simple'
        )
        if clip > 0:
            output = '\n\n'.join(output.split('\n\n')[:-clip])
        return output

    async def create_players(self, scenario, max_players=None, query=None,
                             others=False, pattern_sep=None, pattern_left=None
                             ):
        if query is None:
            query = 'List the key players in this scenario, separated by semicolons.'
        if pattern_sep is None:
            pattern_sep = r'[\.\,;\n0-9]+'
        if pattern_left is None:
            pattern_left = ' ()-'
        template = 'Scenario: {scenario}\n\nQuestion: {query}\n\nAnswer: '
        variables = {'scenario': scenario, 'query': query}
        output = await self.return_output(
            template=template, variables=variables
        )
        names = re.split(pattern_sep, output)
        names = [name.lstrip(pattern_left).rstrip() for name in names]
        names = [name for name in names if len(name) > 0]
        if max_players is None:
            player_names = names
            other_names = []
        else:
            player_names = names[:max_players]
            other_names = names[max_players:]
        players = [Player(llm=self.llm, name=name, persona=name)
                   for name in player_names]
        if not others:
            return players
        else:
            return players, other_names

    async def create_inject(self, history=None, query=None):
        if query is None:
            raise Exception('Query required to create inject.')
        output = await self.return_output(
            history=history,
            query=query, query_format='oneline', query_subtitle='Narrator'
        )
        return output


class Team(Stateful):
    def __init__(self, name='Anonymous', leader=None, members=None):
        self.name = name
        self.leader = leader
        self.members = members
        self.history = History()

    async def respond(self, history=None, query=None, mc=None, short=False):
        member_responses = History()
        for member in self.members:
            if verbose >= 2:
                print('\n### ' + member.name)
            member_responses.add(member.name, await member.respond(
                history=history, query=query, mc=mc, short=short))
        if verbose >= 2:
            print('\n### Leader: ' + self.leader.name)
        leader_response = await self.leader.synthesize(
            history=history, responses=member_responses, query=query, mc=mc)
        return leader_response

    async def synthesize(self, history=None, responses=None, query=None,
                         mc=None):
        return await self.leader.synthesize(
            history=history, responses=responses, query=query, mc=mc)

    def chat(self, history=None):
        return self.leader.chat(history=history)

    def info(self, offset=0):
        print(' ' * offset + 'Team:', self.name)
        print(' ' * offset + '  Leader:', self.leader.name)
        print(' ' * offset + '  Members:', [member.name
                                            for member in self.members])
        if verbose >= 2:
            for member in self.members:
                member.info(offset=offset+2)


class Player(Intelligent, Stateful, RAG):
    def __init__(self, llm=None, name='Anonymous', kind='ai', persona=None,
                 loader=None, chunk_size=None, chunk_overlap=None, count=None,
                 rag_llm=None, ioid=None, iodict=None, presets=None):
        self.llm = llm
        self.name = name
        self.kind = kind
        self.persona = persona
        self.ioid = ioid
        self.iodict = iodict
        self.presets = presets
        self.history = History()
        self.setup_kind(self.kind)

        if loader is not None:
            self.rag_llm = rag_llm if rag_llm is not None else self.llm
            self.rag_init(loader, chunk_size, chunk_overlap, count)
            self.rag = True
        else:
            self.rag = None

    async def respond(self, history=None, query=None, reminder=2, mc=None,
                      short=False):
        if query is None:
            query = 'What action or actions do you take in response?'
        bind = {'stop': ['\n\n']} if short else {'stop': ['Narrator:']}
        output = await self.return_output(
            bind=bind,
            name=self.name,
            persona=self.persona, reminder=reminder,
            rag=self.rag,
            history=history,
            query=query
        )
        if mc is not None:
            output = await self.multiple_choice(query, output, mc)
        return output

    async def synthesize(self, history=None, responses=None, query=None,
                         mc=None):
        if query is None:
            responses_intro = 'These are the actions your team members recommend you take in response'
            synthesize_query = 'Combine the recommended actions given above'
        else:
            responses_intro = 'These are the responses from your team members'
            synthesize_query = 'Combine the responses given above'
        output = await self.return_output(
            name=self.name,
            persona=self.persona,
            history=history,
            responses=responses, responses_intro=responses_intro,
            query=synthesize_query, query_format='oneline'
        )
        if mc is not None:
            output = await self.multiple_choice(query, output, mc)
        return output

    def chat(self, history=None):
        name = self.name
        persona = self.persona
        rag = self.rag
        return self.chat_backend(
            name=name, persona=persona, rag=rag, history=history)

    def info(self, offset=0):
        print(' ' * offset + 'Player:', self.name)
        print(' ' * offset + '  Type:', self.kind)
        print(' ' * offset + '  Persona:', self.persona)
