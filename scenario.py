#!/usr/bin/env python3

import langchain
import langchain.llms
import langchain.prompts
import langchain.chains
import langchain.callbacks

def main():
    template = 'You are the leader of the country of {nation}.  What is a major challenge facing your country?'
    #prompt = langchain.prompts.PromptTemplate.from_template(template)
    prompt = langchain.prompts.PromptTemplate(template=template, input_variables=['nation'])
    prompt.format(nation='Poland')

    llm_openai = langchain.llms.OpenAI()
    #mp = '/home/scenario/wdata/llm/gpt4all/mistral-7b-openorca.Q4_0.gguf'
    #mp = '/home/scenario/wdata/llm/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf'
    #mp = '/home/scenario/wdata/llm/gpt4all/gpt4all-falcon-q4_0.gguf'
    mp = '/home/scenario/wdata/llm/gpt4all/mpt-7b-chat-q4_0.gguf'
    cbm = langchain.callbacks.manager.CallbackManager([langchain.callbacks.streaming_stdout.StreamingStdOutCallbackHandler()])
    llm_llamacpp = langchain.llms.LlamaCpp(
        model_path=mp,
        n_gpu_layers=1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=cbm,
        verbose=True
    )
    llm = llm_llamacpp
    #print(llm.predict(prompt.format_prompt(nation='Papua New Guinea').to_messages()[0].content))

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
