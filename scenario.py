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

def main():
    #template = 'You are the leader of the country of {nation}.  What is a major challenge facing your country?'
    template = """Question: You are the leader of the country of {nation}.
    What is a major challenge facing your country?

    Answer: """

    #prompt = langchain.prompts.PromptTemplate.from_template(template)
    prompt = langchain.prompts.PromptTemplate(template=template, input_variables=['nation'])
    prompt.format(nation='Poland')

    # LLM
    choice = ['openai', 'llamacpp', 'huggingface'][2]
    if choice == 'openai':

        # LLM: OpenAI (Cloud)
        llm_openai = langchain.llms.OpenAI(
            model_name='text-davinci-003'
        )
        llm = llm_openai

    elif choice == 'llamacpp':

        # LLM: llama.cpp (Local)
        mps = [
            '/home/scenario/wdata/llm/gpt4all/mistral-7b-openorca.Q4_0.gguf',
            '/home/scenario/wdata/llm/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf',
            '/home/scenario/wdata/llm/gpt4all/gpt4all-falcon-q4_0.gguf',
            '/home/scenario/wdata/llm/gpt4all/mpt-7b-chat-q4_0.gguf',
        ]
        mp = mps[1]
        cbm = langchain.callbacks.manager.CallbackManager([langchain.callbacks.streaming_stdout.StreamingStdOutCallbackHandler()])
        llm_llamacpp = langchain.llms.LlamaCpp(
            model_path=mp,
            n_gpu_layers=9999,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=cbm,
            verbose=False,
        )
        llm = llm_llamacpp

    elif choice == 'huggingface':

        # Model Source: Hugging Face (Local)
        #model_path = 'mosaicml/mpt-7b-chat'
        #model_path = '/home/scenario/wdata/mosaicml/mpt-7b-chat'
        model_path = '/home/scenario/wdata/mosaicml/mpt-30b-chat'
        #model_path = '/home/scenario/wdata/llm/cerebras/Cerebras-GPT-13B'

        device = torch.device('cuda:' + str(torch.cuda.current_device())
                              if torch.cuda.is_available() else 'cpu')
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            init_device='cuda',
            learned_pos_emb=False,
            max_seq_len=2500,
        )
        #config.attn_config['attn_impl'] = 'torch'

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        model.to(device)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        class StopOnTokens(transformers.StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor,
                         scores: torch.FloatTensor,
                         **kwargs: typing.Any) -> bool:
                for stop_id in tokenizer.convert_tokens_to_ids(
                        ['<|endoftext|>', '<|im_end|>']):
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

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

        """
            #return_full_text=True,
            #temperature=0.1,
            #top_p=0.15,
            #top_k=0,
            #repetition_penalty=1.1,
        """

        llm_huggingface = langchain.llms.HuggingFacePipeline(
            pipeline=pipeline,
        )
        llm = llm_huggingface

    print(llm.predict(prompt.format_prompt(nation='Papua New Guinea').to_messages()[0].content))

    chain = langchain.chains.LLMChain(
        prompt=prompt,
        llm=llm
    )
    print(chain.run(nation='Panama'))
    inputs = [{'nation': 'Palau'},
              {'nation': 'Paraguay'}]
    print(chain.generate(inputs))

if __name__ == '__main__':
    main()
