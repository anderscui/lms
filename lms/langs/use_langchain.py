# coding=utf-8
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lms.langs.utils import get_deepseek_v3_llm


llm = get_deepseek_v3_llm()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are an AI expert'),
    ('user', '{input}')
])

output_parser = StrOutputParser()

# 链式调用
chain = prompt | llm | output_parser

result = chain.invoke({'input': 'please write an article about AI for me, no more than 100 words.'})
print(result)

chain2 = prompt | llm
result2 = chain2.invoke({'input': 'please write an article about AI for me, no more than 100 words.'})
print(result2)
