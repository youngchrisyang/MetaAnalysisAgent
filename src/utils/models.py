import os,json
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField, RunnableConfig
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from .initialization import initialize_env

initialize_env()

## OPENAI
GPT4O_LANGCHAIN = ChatOpenAI(
        model_name='gpt-4o-2024-05-13',
        #model_name='gpt-4o-2024-08-06',
        temperature=0,
    )

GPT4O_LANGCHAIN_NEW = ChatOpenAI(
        #model_name='gpt-4o-2024-05-13',
        model_name='gpt-4o-2024-08-06',
        temperature=0,
    )

GPT4OMINI_LANGCHAIN = ChatOpenAI(
        model_name='gpt-4o-mini-2024-07-18',
        temperature=0,
    )

GPT4TURBO_LANGCHAIN = ChatOpenAI(
        model_name='gpt-4-turbo-2024-04-09',
        temperature=0,
    )

GPT35TURBO_LANGCHAIN = ChatOpenAI(
        model_name='gpt-3.5-turbo-1106',
        temperature=0,
    )

## ANTHROPIC
CLAUDE3_LANGCHAIN = ChatAnthropic(
    model='claude-3-opus-20240229',
    temperature=0
    )

CLAUDE35_SONNET_LANGCHAIN =  ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0
    )

## Standard Fallback Model
LLM_STANDARD_FALLBACK = GPT4O_LANGCHAIN.with_fallbacks(
    [
        GPT4O_LANGCHAIN_NEW,
        CLAUDE35_SONNET_LANGCHAIN,
     ]
)

def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings()