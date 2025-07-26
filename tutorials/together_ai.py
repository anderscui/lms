# coding=utf-8
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import ChatTogether

load_dotenv()
# print(os.getenv('TOGETHER_API_KEY'))

# models
# meta-llama/Llama-3.3-70B-Instruct-Turbo
# mistralai/Mistral-7B-Instruct-v0.2
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
# deepseek-ai/DeepSeek-V3
# deepseek-ai/DeepSeek-R1
# Qwen/Qwen2.5-72B-Instruct-Turbo
# Qwen/Qwen2.5-7B-Instruct-Turbo

# embedding
# togethercomputer/m2-bert-80M-2k-retrieval
# togethercomputer/m2-bert-80M-8k-retrieval
# togethercomputer/m2-bert-80M-32k-retrieval
# BAAI/bge-base-en-v1.5
# BAAI/bge-large-en-v1.5
# WhereIsAI/UAE-Large-V1


llm = ChatTogether(
    model='deepseek-ai/DeepSeek-V3',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# messages = [
#     ('system', 'You are a helpful assistant that translates english to chinese. Translate the user input.'),
#     ('human', 'I love machine learning.')
# ]
# result = llm.invoke(messages)
# print(result)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         # ('system', 'You are a professional translator, translate the user input from {input_lang} to {output_lang}.'),
#         ('system', '你是一个很伟大的译者，请将用户输入从 {input_lang} 翻译为 {output_lang}.'),
#         ('human', '{input}')
#     ]
# )
#
# chain = prompt | llm
#
# translated_msg = chain.invoke({
#     'input_lang': 'English',
#     'output_lang': 'Chinese',
#     'input': 'I love programming.',
# })
# print(translated_msg)

prompt = ChatPromptTemplate.from_messages(
    [
        # ('system', 'You are a professional translator, translate the user input from {input_lang} to {output_lang}.'),
        ('system', '你是一个很出色的编辑，请找出下面文字中的语法和拼写错误.'),
        ('human', '{input}')
    ]
)

chain = prompt | llm

translated_msg = chain.invoke({
    'input': 'I love programmming. 我和她渡过了一段愉快的时间。',
})
print(translated_msg)
