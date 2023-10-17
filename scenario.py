#!/usr/bin/env python3

import langchain
import langchain.llms
import langchain.prompts

def main():
    template = 'You are the leader of the country of {nation}.  What is a major challenge facing your country?'
    #prompt = langchain.prompts.PromptTemplate.from_template(template)
    prompt = langchain.prompts.PromptTemplate(template=template, input_variables=['nation'])
    prompt.format(nation='Poland')

    llm_openai = langchain.llms.OpenAI()
    llm = llm_openai
    print(llm.predict(prompt.format_prompt(nation='Papua New Guinea').to_messages()[0].content))

if __name__ == '__main__':
    main()
