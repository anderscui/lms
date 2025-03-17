# coding=utf-8
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate

# Chat
# Message type: use one of 'human', 'user', 'ai', 'assistant', or 'system'.
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are an AI assistant, your name is {name}.'),
    ('human', 'hello'),
    ('ai', 'I am good, thanks.'),
    ('human', '{user_input}'),
])

messages = chat_template.format_messages(name='Eliza', user_input='What is your name?')
print(messages)

# Chat by Message instances
chat_template2 = ChatPromptTemplate.from_messages([
    SystemMessage(content='You are an AI assistant, your name is {name}.'),
    HumanMessagePromptTemplate.from_template('{text}')
])

messages2 = chat_template2.format_messages(name='Eliza', text='What is your name?')
print(messages2)

# Simple String Prompt
prompt_template = PromptTemplate.from_template(
    'Show me a {adj} joke about {topic}.'
)

simple_prompt = prompt_template.format(adj='fantastic', topic='programmer')
print(simple_prompt)

# MessagesPlaceholder
