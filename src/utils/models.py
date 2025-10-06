import os,json
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import ConfigurableField, RunnableConfig
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from .initialization import initialize_env

initialize_env()

## OPENAI

GPT41_LANGCHAIN = ChatOpenAI(
    model_name='gpt-4.1-2025-04-14',
    temperature=0,
)

GPT5_LANGCHAIN = ChatOpenAI(
    model_name='gpt-5-2025-08-07',
    temperature=1,
)

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
CLAUDE4_SONNET_LANGCHAIN = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=64000
)

CLAUDE45_SONNET_LANGCHAIN = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0,
    max_tokens=64000
)

## Standard Fallback Model
LLM_GPT41_FALLBACK = GPT41_LANGCHAIN.with_fallbacks(
    [
        CLAUDE45_SONNET_LANGCHAIN,
        GPT5_LANGCHAIN,
     ]
)

LLM_GPT5_FALLBACK = GPT5_LANGCHAIN.with_fallbacks(
    [
        CLAUDE45_SONNET_LANGCHAIN,
        GPT41_LANGCHAIN
    ]
)

LLM_CLAUDE45_FALLBACK = CLAUDE45_SONNET_LANGCHAIN.with_fallbacks(
    [
        GPT41_LANGCHAIN,
        GPT5_LANGCHAIN
    ]
)

LLM_CLAUDE4_FALLBACK = CLAUDE4_SONNET_LANGCHAIN.with_fallbacks(
    [
        GPT41_LANGCHAIN,
        GPT5_LANGCHAIN
    ]
)

def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings()