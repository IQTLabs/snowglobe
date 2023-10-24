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

class LLM():
    def __init__(self, source_name=None, model_name=None):

        # Select large language model
        default_source_name = 'huggingface'
        default_model_name = 'mpt-7b-chat'
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
                'mpt-7b-chat_auto' : 'mosaicml/mpt-7b-chat',
                'mpt-7b-chat' : '/home/scenario/wdata/mosaicml/mpt-7b-chat',
                'mpt-30b-chat' : '/home/scenario/wdata/mosaicml/mpt-30b-chat',
                'Cerebras-GPT-13B' : '/home/scenario/wdata/llm/cerebras/Cerebras-GPT-13B',
            }
        }
        if source_name is not None and model_name is not None:
            self.source_name = source_name
            self.modeel_name = model_name
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
                n_gpu_layers=9999,
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
            #config.attn_config['attn_impl'] = 'torch'

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
