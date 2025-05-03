import os
from typing import List, Literal, Optional, Dict
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from pydantic import BaseModel,Field

from src.utils.models import GPT4O_LANGCHAIN_NEW, GPT41_LANGCHAIN
from .data_types import (
    PaperMetaInfo, 
    SampleBasicInfo, 
    VariableInfoInSample, 
    CorrelationInfoInSample
)

class IsPaperRelevant(BaseModel):
    is_relevant: bool = Field(description="Whether the paper is relevant to the research topic")
    reason: str = Field(description="The reason why the paper is relevant or not relevant to the research topic")

PAPER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to determine if a paper is relevant to a research topic."),
    ("human", """
    I need to determine if this paper is relevant to a research topic.
    
    The research topic is to study the relationship between {independent_variables} and {dependent_variable}.
    
    Please respond with "True" if the paper is relevant to the research topic, and "False" if it is not.
    Be generous in determining if a paper is relevant.

    Paper Content:
    {paper_content}
    """),
    ("human", """
    Please follow these instructions when determining if the paper is relevant: 
    {instructions}
    """),
])

def get_is_paper_relevant_chain(llm = GPT41_LANGCHAIN):
    return PAPER_RELEVANCE_PROMPT | llm.with_structured_output(IsPaperRelevant)

EXTRACT_PAPER_META_INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to extract paper meta information."),
    ("human", """
    I need you to extract the following meta information from this paper:
    {fields_to_extract}
    
    Paper Content:
    {paper_content}
    """),
    ("human", """
    Please follow these instructions when extracting paper meta information:
    {instructions}
    """),
])

def get_extract_paper_meta_info_chain(llm = GPT41_LANGCHAIN):
    # Generate the fields to extract based on the PaperMetaInfo model
    fields = []
    for field_name in PaperMetaInfo.model_fields:
        field_obj = PaperMetaInfo.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_PAPER_META_INFO_PROMPT | llm.with_structured_output(PaperMetaInfo)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain


class ExtractSamplesBasicInfo(BaseModel):
    sample_info: List[SampleBasicInfo] = Field(
        description="""
        First determine how many different samples were used in the paper.
        Then, for each sample, extract the basic information about the sample.
        """
    )

EXTRACT_SAMPLES_BASIC_INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to extract sample information from research papers."),
    ("human", """
    Given this research paper, I need you to extract the sample information.
    
    For each sample, please extract the following information:
    {fields_to_extract}
    
    Research Paper:
    {paper_content}
    """),
    ("human", """
    Please follow these additional instructions when extracting sample information:
    {instructions}
    """),
])

def get_extract_samples_basic_info_chain(llm = GPT41_LANGCHAIN):
    # Generate the fields to extract based on the SampleBasicInfo model
    fields = []
    for field_name in SampleBasicInfo.model_fields:
        field_obj = SampleBasicInfo.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_SAMPLES_BASIC_INFO_PROMPT | llm.with_structured_output(ExtractSamplesBasicInfo)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain


class ExtractVariablesInfoFromSample(BaseModel):
    variables_info: List[VariableInfoInSample] = Field(description="The desired information for each variable")
    correlations_info: List[CorrelationInfoInSample] = Field(description="The correlations between any pairs of variables")

EXTRACT_VARIABLES_INFO_FROM_SAMPLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to extract variable information from research papers."),
    ("human", """
    I need you to extract variable information for a specific sample from this research paper.
    
    Instructions:
    1. You will be given the sample description - please complete the task by only using content related to the sample.
    2. For each of the given variables, extract the basic information about the variable related to the sample.
    
    For each variable, please extract the following information:
    {variable_fields_to_extract}
    
    3. For each pair of variables, determine if there is information on correlation between them. If so, extract the correlation information:
    {correlation_fields_to_extract}
    
    Sample Description:
    {sample_description}
    
    Variables: {variables}
    
    Paper Content:
    {paper_content}
    """),
    ("human", """
    Please follow these additional instructions when extracting variable information:
    {instructions}
    """),
])

def get_extract_variables_info_from_sample_chain(llm = GPT41_LANGCHAIN):
    # Generate the fields to extract for variables
    variable_fields = []
    for field_name in VariableInfoInSample.model_fields:
        field_obj = VariableInfoInSample.model_fields[field_name]
        description = field_obj.description
        variable_fields.append(f"- {field_name}: {description}")
    
    variable_fields_to_extract = "\n".join(variable_fields)
    
    # Generate the fields to extract for correlations
    correlation_fields = []
    for field_name in CorrelationInfoInSample.model_fields:
        field_obj = CorrelationInfoInSample.model_fields[field_name]
        description = field_obj.description
        correlation_fields.append(f"- {field_name}: {description}")
    
    correlation_fields_to_extract = "\n".join(correlation_fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_VARIABLES_INFO_FROM_SAMPLE_PROMPT | llm.with_structured_output(ExtractVariablesInfoFromSample)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["variable_fields_to_extract"] = variable_fields_to_extract
        inputs["correlation_fields_to_extract"] = correlation_fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain
