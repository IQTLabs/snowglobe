#!/usr/bin/env python3

import os
import re
import yaml
import torch
import triton
import typing
import transformers

import langchain
import langchain.llms
import langchain.chains
import langchain.schema
import langchain.prompts
import langchain.callbacks

class LLM():
    def __init__(self, source_name=None, model_name=None, menu=None):

        # Select large language model
        default_source_name = 'llamacpp'
        default_model_name = 'mistral-7b-openorca'
        default_menu = os.path.join(os.path.split(__file__)[0], 'llms.yaml')

        self.menu = menu if menu is not None else default_menu
        model_paths = yaml.safe_load(open(self.menu, 'r'))
        if source_name is not None and model_name is not None:
            self.source_name = source_name
            self.model_name = model_name
        else:
            self.source_name = default_source_name
            self.model_name = default_model_name
        self.model_path = model_paths[self.source_name][self.model_name]

        if self.source_name == 'openai':

            # Model Source: OpenAI (Cloud)
            cbm = langchain.callbacks.manager.CallbackManager(
                [langchain.callbacks.streaming_stdout\
                 .StreamingStdOutCallbackHandler()])
            self.llm = langchain.llms.OpenAI(
                model_name=self.model_name,
                callback_manager=cbm,
                streaming=True,
            )
            self.bound = {}

        elif self.source_name == 'llamacpp':

            # Model Source: llama.cpp (Local)
            cbm = langchain.callbacks.manager.CallbackManager(
                [langchain.callbacks.streaming_stdout\
                 .StreamingStdOutCallbackHandler()])
            self.llm = langchain.llms.LlamaCpp(
                model_path=self.model_path,
                n_gpu_layers=-1,
                max_tokens=1000,
                n_batch=512,
                n_ctx=8192,
                f16_kv=True,
                callback_manager=cbm,
                verbose=False,
            )
            self.bound = {'stop': '##'}

        elif self.source_name == 'huggingface':

            # Model Source: Hugging Face (Local)
            device = torch.device('cuda:' + str(torch.cuda.current_device())
                                  if torch.cuda.is_available() else 'cpu')
            config = transformers.AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                init_device='cuda',
                learned_pos_emb=False,
                max_seq_len=2500,
            )
            # config.attn_config['attn_impl'] = 'torch'

            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            model.eval()
            model.to(device)

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_path)

            class StopOnTokens(transformers.StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor,
                             scores: torch.FloatTensor,
                             **kwargs: typing.Any) -> bool:
                    for stop_id in tokenizer.convert_tokens_to_ids(
                            ['<|endoftext|>', '<|im_end|>']):
                        if input_ids[0][-1] == stop_id:
                            return True
                    return False
            stopping_criteria = transformers.StoppingCriteriaList(
                [StopOnTokens()])

            streamer = transformers.TextStreamer(
                tokenizer, skip_prompt=True, skip_special=True)

            pipeline = transformers.pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=2048,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )

            self.llm = langchain.llms.HuggingFacePipeline(
                pipeline=pipeline,
            )
            self.bound = {}


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
    def str(self, name=None):
        return '\n\n'.join(['' \
                        + ('You' if entry['name'] == name else entry['name'])
                        + ':\n\n' + entry['text']
                        for entry in self.entries])
    def textonly(self):
        return '\n\n'.join([entry['text'] for entry in self.entries])
    def copy(self):
        history_copy = History()
        history_copy.entries = self.entries.copy()
        return history_copy


class Control():
    def __init__(self, llm_source_name=None, llm_model_name=None):
        self.llm = LLM(source_name=llm_source_name, model_name=llm_model_name)
        self.history = History()

    def run(self):
        raise Exception('! Override this method in the subclass for your specific scenario.')

    def header(self, title, h=0, width=80):
        print()
        if h == 0:
            print('+-' + '-' * min(len(title), width) + '-+')
            print('| ' + title + ' |')
            print('+-' + '-' * min(len(title), width) + '-+')
        elif h == 1:
            print('-' * min(len(title), width))
            print(title)
            print('-' * min(len(title), width))
        else:
            print(title)

    def record_narration(self, narration):
        self.history.add('Narrator', narration)

    def record_response(self, player_name, player_response):
        self.history.add(player_name, player_response)

    def adjudicate(self, history=None, responses=None, query=None,
                   nature=True, timeframe='week', verbose=0):
        if query is None:
            #query = 'This is an example of what happens next as a result of these plans.'
            if nature:
                query = 'This is what happens in the next ' + timeframe \
                    + ' due to these plans. Include unexpected developments.'
            else:
                query = 'This is what happens in the next ' + timeframe \
                    + ' due to these plans.'

        # Define template and included variables
        template = ''
        variables = {}
        if history is not None:
            template += '### This is what has happened so far:\n\n{history}\n\n'
            variables['history'] = history.textonly()
        if responses is not None:
            #template += '### These are the actions undertaken by each person or group:\n\n{responses}\n\n'
            template += '### These are the plans for each person or group:\n\n{responses}\n\n'
            variables['responses'] = responses.str()
        #template += '### Question: {query}\n\n### Answer:\n\n'
        template += '### {query}:\n\n'
        variables['query'] = query

        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        chain = prompt | self.llm.llm.bind(**self.llm.bound)
        if verbose >= 2:
            print(prompt.format(**variables))
            print()
        output = chain.invoke(variables).strip()
        print()
        if False:
            print('\n### Summary\n')
            chain_novel = novel(self.llm.llm.bind(**self.llm.bound))
            output = chain_novel.invoke({
                'history': history.textonly(),
                'news': output,
            }).strip()
            print()
        return output

    def assess(self, history, query):
        template = '### This is what happened.\n\n{history}\n\n### Question:\n\n{query}\n\n### Answer:\n\n'
        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=['history', 'query'],
        )
        chain = prompt | self.llm.llm.bind(**self.llm.bound)
        output = chain.invoke({
            'history': history.str(), 'query': query}).strip()
        print()
        return output

    def chat(self, history=None, verbose=1):
        if history is None:
            history = self.history
        chatlog = History()
        nb = 2
        if verbose >= 1:
            instructions = 'Start typing to discuss the simulation, or press Enter twice to exit.'
            self.header(instructions, h=1)
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
            chatlog.add('User', usertext)

            # Create template
            template = '### You are the Control (a.k.a. moderator) of a simulated scenario.\n\n'
            variables = {}
            if len(history.entries) >= 0:
                template += '### This is what happened.\n\n{history}\n\n'
                variables['history'] = history.str()
            template += '### This is a conversation about what happened.\n\n{chat}\n\nControl:\n\n'
            variables['chat'] = chatlog.str()

            # Get LLM response
            prompt = langchain.prompts.PromptTemplate(
                template=template,
                input_variables=list(variables.keys()),
            )
            chain = prompt | self.llm.llm.bind(**self.llm.bound).bind(
                stop=['User:', 'Control:', 'Narrator:'])
            if verbose >= 2:
                print(prompt.format(**variables))
                print()
            output = chain.invoke(variables).strip()
            print()
            print()
            chatlog.add('Control', usertext)

    def create_scenario(self, query=None, clip=0):
        if query is None:
            raise Exception('Query required to create scenario.')
        prompt = langchain.prompts.PromptTemplate(
            template='Question: {query}\n\nAnswer: ',
            input_variables=['query'],
        )
        chain = prompt | self.llm.llm.bind(**self.llm.bound)
        output = chain.invoke({'query': query}).strip()
        print()
        if clip > 0:
            output = '\n\n'.join(output.split('\n\n')[:-clip])
        return output

    def create_players(self, scenario, max_players=None, query=None,
                       pattern_sep=None, pattern_left=None):
        if query is None:
            query = 'List the key players in this scenario, separated by semicolons.'
        if pattern_sep is None:
            pattern_sep = '[\.\,;\n0-9]+'
        if pattern_left is None:
            pattern_left = ' ()-'
        prompt = langchain.prompts.PromptTemplate(
            template='Scenario: {scenario}\n\nQuestion: {query}\n\nAnswer: ',
            input_variables=['scenario', 'query'],
        )
        chain = prompt | self.llm.llm.bind(**self.llm.bound)
        output = chain.invoke({'scenario': scenario, 'query': query}).strip()
        print()
        names = re.split(pattern_sep, output)
        names = [name.lstrip(pattern_left).rstrip() for name in names]
        names = [name for name in names if len(name) > 0]
        print(names)
        if max_players is not None:
            names = names[:max_players]
        players = [Player(llm=self.llm, name=name, persona=name)
                   for name in names]
        return players

    def create_inject(self, history=None, query=None):
        if query is None:
            raise Exception('Query required to create inject.')

        template = ''
        variables = {}
        if history is not None:
            template += '### This is what has happened so far:\n\n{history}\n\n'
            variables['history'] = history.str()
        #template += '### {query}:\n\n'
        #template += 'Narrator:\nUnexpectedly, '
        template += '### {query}:\n\nNarrator:\n'
        variables['query'] = query

        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        chain = prompt | self.llm.llm.bind(**self.llm.bound).bind(
                stop=['\n\n'])
        output = chain.invoke(variables).strip()
        print()
        return output


class Team():
    def __init__(self, name='Anonymous', leader=None, members=None):
        self.name = name
        self.leader = leader
        self.members = members
        self.history = History()

    def respond(self, history=None, query=None, verbose=0):
        member_responses = History()
        for member in self.members:
            if verbose >= 1:
                print('\n### ' + member.name)
            member_responses.add(member.name, member.respond(
                history=history, query=query, verbose=verbose))
        if verbose >= 1:
            print('\n### Leader: ' + self.leader.name)
        leader_response = self.leader.synthesize(
            history=history, responses=member_responses, query=query,
            verbose=verbose)
        return leader_response

    def synthesize(self, history=None, responses=None, query=None, verbose=0):
        return self.leader.synthesize(
            history=history, responses=responses, query=query, verbose=verbose)

    def info(self, verbose=0, offset=0):
        print(' ' * offset + 'Team:', self.name)
        print(' ' * offset + '  Leader:', self.leader.name)
        print(' ' * offset + '  Members:', [member.name
                                            for member in self.members])
        if verbose >= 1:
            for member in self.members:
                member.info(verbose=verbose, offset=offset+2)


class Player():
    def __init__(self, llm, name='Anonymous', kind='ai', persona=None):
        self.llm = llm
        self.name = name
        self.kind = kind
        self.persona = persona
        self.history = History()

    def respond(self, history=None, query=None, max_tries=64, verbose=0):
        persona = self.persona
        if query is None:
            query = 'What action or actions do you take in response?'

        # Define template and included variables
        template = ''
        variables = {}
        if persona is not None:
            template += '### You are {persona}.\n\n'
            variables['persona'] = persona
        if history is not None:
            template += '### This is what has happened so far:\n\n{history}\n\n'
            variables['history'] = history.str(name=self.name)
        template += '### Question:\n\n{query}'
        variables['query'] = query
        if persona is not None:
            template += ' (Remember, you are {persona}.)'
        template += '\n\n### Answer:\n\n'

        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        llm = self.llm.llm.bind(**self.llm.bound)
        chain = prompt | llm
        if verbose >= 2:
            print(prompt.format(**variables))
            print()
        for i in range(max_tries):
            output = chain.invoke(variables).strip()
            if len(output) > 0:
                break
        print()
        if False:
            print('\n###\n')
            chain_desub = desubjunctifier(llm)
            output = chain_desub.invoke(input=output).strip()
        return output

    def synthesize(self, history=None, responses=None, query=None, verbose=0):
        persona = self.persona

        # Define template and included variables
        template = ''
        variables = {}
        if persona is not None:
            template += '### You are {persona}.\n\n'
            variables['persona'] = persona
        if history is not None:
            template += '### This is what has happened so far:\n\n{history}\n\n'
            variables['history'] = history.str(name=self.name)
        if responses is not None:
            if query is None:
                template += '### These are the actions your team members recommend you take in response:\n\n{responses}\n\n'
            else:
                template += '### These are the responses from your team members:\n\n{responses}\n\n'
            variables['responses'] = responses.str(name=self.name)
        # template += '### What action or actions do you take in response?:\n\n'
        if query is None:
            template += '### Combine the recommended actions given above:\n\n'
        else:
            template += '### Combine the responses given above:\n\n'
        # if persona is not None:
        #     template += ' (Remember, you are {persona}.)'

        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        chain = prompt | self.llm.llm.bind(**self.llm.bound)
        if verbose >= 2:
            print(prompt.format(**variables))
            print()
        output = chain.invoke(variables).strip()
        print()
        return output

    def info(self, verbose=0, offset=0):
        print(' ' * offset + 'Player:', self.name)
        print(' ' * offset + '  Type:', self.kind)
        print(' ' * offset + '  Persona:', self.persona)


def novel(llm):
    template = '### History:\n\n{history}\n\n\n### News:\n\n{news}\n\n### In a short paragraph, what\'s the most important information appearing in the news but not in the history?  Do not use phrases similar to "most important news" in your answer:'
    prompt = langchain.prompts.PromptTemplate(
        template=template,
        input_variables=['history', 'news'],
    )
    chain = prompt | llm
    return chain


def desubjunctifier(llm):
    examples = [{
        'input': "A) We would undertake an audit."
        "\n\nB) I could send a message to the ambassador, stating our intentions."
        "\n\nC) I implement a response plan."
        "\n\nD) After that, immediately deploy the medics. Furnish all needed supplies.",
        'output': "A) We undertake an audit."
        "\n\nB) I send a message to the ambassador, stating our intentions."
        "\n\nC) I implement a response plan."
        "\n\nD) After that, we immediately deploy the medics. We furnish all needed supplies.",
    },{
        'input': "Continue to investigate the source of the problems. I will refrain from making accusations. We should optimize the allocation of resources to different departments. This should not be thrown away.",
        'output': "I continue to investigate the source of the problems. I refrain from making accusations. We optimize the allocation of resources to different departments. This is not thrown away.",
    }]
    example_template = 'Input:\n"""\n{input}\n"""\nOutput:\n"""\n{output}\n"""'
    example_prompt = langchain.prompts.PromptTemplate(
        template=example_template,
        input_variables=['input', 'output']
    )
    prefix = 'Change every Input sentence into a first-person present indicative statement.'
    suffix = 'Input:\n"""\n{input}\n"""\nOutput:\n"""\n'
    prompt = langchain.prompts.FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=['input']
    )
    chain = {'input': langchain.schema.runnable.RunnablePassthrough()} | prompt | llm
    return chain
