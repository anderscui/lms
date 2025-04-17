# coding=utf-8
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lms.langs.utils import get_deepseek_v3_llm


llm = get_deepseek_v3_llm()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a storyteller'),
    ('user', """"Current conversation: {chat_history}
                 
                 {input}""")
])

memory = ConversationBufferMemory(memory_key='chat_history')

chain = LLMChain(llm=llm,
                 prompt=prompt,
                 memory=memory)

result = chain.invoke({'input': 'Hi, my name is Anders. What is 1+1?'})
print(result)

result2 = chain.invoke(({'input': 'What is my name?'}))
print(result2)
