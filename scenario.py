#!/usr/bin/env python3

import torch
import triton
import typing
import transformers

import langchain
import langchain.llms
import langchain.prompts
import langchain.chains
import langchain.callbacks
import langchain.memory

class LLM():
    def __init__(self, source_name=None, model_name=None):

        # Select large language model
        default_source_name = 'llamacpp'
        default_model_name = 'mistral-7b-openorca'
        model_paths = {
            'openai' : {
                'text-davinci-003' : '',
            },
            'llamacpp' : {
                'mistral-7b-openorca' : '/home/scenario/wdata/llm/gpt4all/mistral-7b-openorca.Q4_0.gguf',
                'mistral-7b-instruct' : '/home/scenario/wdata/llm/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf',
                'gpt4all-falcon' : '/home/scenario/wdata/llm/gpt4all/gpt4all-falcon-q4_0.gguf',
                'mpt-7b-chat' : '/home/scenario/wdata/llm/gpt4all/mpt-7b-chat-q4_0.gguf',
            },
            'huggingface' : {
                'mpt-7b' : '/home/scenario/wdata/mosaicml/mpt-7b',
                'mpt-7b-chat' : '/home/scenario/wdata/mosaicml/mpt-7b-chat',
                'mpt-7b-instruct': '/home/scenario/wdata/mosaicml/mpt-7b-instruct',
                'mpt-7b-storywriter': '/home/scenario/wdata/mosaicml/mpt-7b-storywriter',
                'mpt-30b-chat' : '/home/scenario/wdata/mosaicml/mpt-30b-chat',
                'Cerebras-GPT-13B' : '/home/scenario/wdata/llm/cerebras/Cerebras-GPT-13B',
            }
        }
        if source_name is not None and model_name is not None:
            self.source_name = source_name
            self.model_name = model_name
        else:
            self.source_name = default_source_name
            self.model_name = default_model_name
        self.model_path = model_paths[self.source_name][self.model_name]

        if self.source_name == 'openai':

            # Model Source: OpenAI (Cloud)
            self.llm = langchain.llms.OpenAI(
                model_name=self.model_name
            )

        elif self.source_name == 'llamacpp':

            # Model Source: llama.cpp (Local)
            cbm = langchain.callbacks.manager.CallbackManager(
                [langchain.callbacks.streaming_stdout\
                 .StreamingStdOutCallbackHandler()])
            self.llm = langchain.llms.LlamaCpp(
                model_path=self.model_path,
                n_gpu_layers=-1,
                n_batch=512,
                n_ctx=2048,
                f16_kv=True,
                callback_manager=cbm,
                verbose=False,
            )

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


class History():
    def __init__(self):
        self.entries = []
    def add(self, name, text):
        self.entries.append({'name': name, 'text': text})
    def str(self, name=None):
        return ''.join([('You' if entry['name'] == name else entry['name'])
                        + ': ' + entry['text'] + '\n'
                        for entry in self.entries])
    def copy(self):
        history_copy = History()
        history_copy.entries = self.entries.copy()
        return history_copy


class Control():
    def __init__(self, llm_source_name=None, llm_model_name=None):
        self.llm = LLM(source_name=llm_source_name, model_name=llm_model_name).llm
        self.public_history = History()

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
        self.public_history.add('Narrator', narration)

    def record_response(self, player_name, player_response):
        self.public_history.add(player_name, player_response)

    def assess(self, history, query):
        template = 'This is what happened.\n{history}\n{query}'
        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=['history', 'query'],
        )
        chain = langchain.chains.LLMChain(
            prompt=prompt,
            llm=self.llm
        )
        output = chain.run(history=history.str(), query=query).strip()
        print()
        return output


class Team():
    def __init__(self, name='Anonymous', leader=None, members=None):
        self.name = name
        self.leader = leader
        self.members = members

    def respond(self, history=None, query=None, verbose=0):
        member_responses = History()
        for member in self.members:
            if verbose >= 1:
                print('\n### ' + member.name)
            member_responses.add(member.name, member.respond(
                history=history, query=query, verbose=verbose))
        leader_response = self.leader.synthesize(
            history=history, responses=member_responses, verbose=verbose)
        return leader_response

    def info(self):
        print('Team:', self.name)
        print('  Leader:', self.leader.name)
        print('  Members:', [member.name for member in self.members])


class Player():
    def __init__(self, llm, name='Anonymous', kind='ai', persona=None):
        self.llm = llm
        self.name = name
        self.kind = kind
        self.persona = persona

    def respond(self, history=None, query=None, verbose=0):
        persona = self.persona

        # Define template and included variables
        template = ''
        variables = {}
        if persona is not None:
            template += 'You are {persona}.\n\n'
            variables['persona'] = persona
        if history is not None:
            template += 'This is what has happened so far.\n{history}\n'
            variables['history'] = history.str(name=self.name)
        if query is not None:
            template += '{query}'
            variables['query'] = query
        else:
            template += 'What action or actions do you take in response?:'
        if persona is not None:
            template += ' (Remember, you are {persona}.)'

        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        chain = langchain.chains.LLMChain(
            prompt=prompt,
            llm=self.llm
        )
        if verbose >= 2:
            print(chain.prompt.format(**variables))
            print()
        output = chain.run(**variables).strip()
        print()
        return output

    def synthesize(self, history=None, responses=None, verbose=0):
        persona = self.persona

        # Define template and included variables
        template = ''
        variables = {}
        if persona is not None:
            template += 'You are {persona}.\n\n'
            variables['persona'] = persona
        if history is not None:
            template += 'This is what has happened so far.\n{history}\n'
            variables['history'] = history.str(name=self.name)
        if responses is not None:
            template += 'These are the actions your team members recommend you take in response.\n{responses}\n'
            variables['responses'] = responses.str(name=self.name)
        template += 'What action or actions do you take in response?:'
        if persona is not None:
            template += ' (Remember, you are {persona}.)'

        prompt = langchain.prompts.PromptTemplate(
            template=template,
            input_variables=list(variables.keys()),
        )
        chain = langchain.chains.LLMChain(
            prompt=prompt,
            llm=self.llm
        )
        if verbose >= 2:
            print(chain.prompt.format(**variables))
            print()
        output = chain.run(**variables).strip()
        print()
        return output

    def info(self):
        print('Player:', self.name)
        print('  Type:', self.kind)
        print('  Persona:', self.persona)
