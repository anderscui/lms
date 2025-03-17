# coding=utf-8
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

load_dotenv()

# together ai chat models
# mistralai/Mistral-7B-Instruct-v0.2
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.3-70B-Instruct-Turbo
# meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
# deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
# deepseek-ai/DeepSeek-V3
# deepseek-ai/DeepSeek-R1
# Qwen/Qwen2.5-72B-Instruct-Turbo
# Qwen/Qwen2.5-7B-Instruct-Turbo


def get_together_llm(model):
    # TODO: adjust config.
    llm = ChatTogether(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    return llm


llm_map = {
    'deepseek_v3': (get_together_llm, 'deepseek-ai/DeepSeek-V3'),
    'deepseek_r1': (get_together_llm, 'deepseek-ai/DeepSeek-R1'),
    'qwen2.5-7b': (get_together_llm, 'Qwen/Qwen2.5-7B-Instruct-Turbo'),
    'qwen2.5-72b': (get_together_llm, 'Qwen/Qwen2.5-72B-Instruct-Turbo'),
    'llama-3.3-70b': (get_together_llm, 'meta-llama/Llama-3.3-70B-Instruct-Turbo'),
    'mistral-7b-instruct': (get_together_llm, 'mistralai/Mistral-7B-Instruct-v0.3'),
}


def get_openai_llm():
    return ChatOpenAI()


def get_deepseek_v3_llm():
    loader, model = llm_map['deepseek_v3']
    return loader(model)


def get_deepseek_r1_llm():
    loader, model = llm_map['deepseek_r1']
    return loader(model)


def get_qwen25_7b_llm():
    loader, model = llm_map['qwen2.5-7b']
    return loader(model)


def get_llama_33_70b_llm():
    loader, model = llm_map['llama-3.3-70b']
    return loader(model)


def get_mistral_7b_llm():
    loader, model = llm_map['mistral-7b-instruct']
    return loader(model)
