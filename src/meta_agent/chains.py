import os
from typing import List, Literal, Optional
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.pydantic_v1 import BaseModel,Field

from src.utils.models import GPT4O_LANGCHAIN_NEW
from .data_types import PaperMetaInfo

class IsPaperRelevant(BaseModel):
    is_relevant: bool

PAPER_RELEVANCE_PROMPT = PromptTemplate.from_template(
    """
    You are a intelligent academic researcher. You are given the first a few pages of a paper, which includes a title, abstract, and a few pages of content.
    Your task is to determine if the paper is relevant to a research topic.

    The research topic is to study the relationship between {independent_variable} and {dependent_variable}.

    Please respond with "True" if the paper is relevant to the research topic, and "False" if it is not.

    Paper Content:
    {paper_content}

    Output
    """
)

def get_is_paper_relevant_chain(llm = GPT4O_LANGCHAIN_NEW):
    return PAPER_RELEVANCE_PROMPT | llm.with_structured_output(IsPaperRelevant)

EXTRACT_PAPER_META_INFO_PROMPT = PromptTemplate.from_template(
    """
    You are a intelligent academic researcher. Your task is to extract the paper meta information.

    Instructions:
    1. You are given part of the paper in "Paper Content".
    2. Fields to extract:
        - paper_id
        - title
        - publication_year
        - journal
        - published_status
        - publication_type
        - num_studies
        - num_samples

    Paper Content:
    {paper_content}
    
    Output:
    """
)

def get_extract_paper_meta_info_chain(llm = GPT4O_LANGCHAIN_NEW):
    return EXTRACT_PAPER_META_INFO_PROMPT | llm.with_structured_output(PaperMetaInfo)