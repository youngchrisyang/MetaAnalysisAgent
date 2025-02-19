import os
from typing import List, Literal, Optional, Dict
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.pydantic_v1 import BaseModel,Field

from src.utils.models import GPT4O_LANGCHAIN_NEW
from .data_types import (
    PaperMetaInfo, 
    SampleBasicInfo, 
    VariableInfoInSample, 
    CorrelationInfoInSample
)

class IsPaperRelevant(BaseModel):
    is_relevant: bool

PAPER_RELEVANCE_PROMPT = PromptTemplate.from_template(
    """
    You are a intelligent academic researcher. You are given the first a few pages of a paper, which includes a title, abstract, and a few pages of content.
    Your task is to determine if the paper is relevant to a research topic.

    The research topic is to study the relationship between {independent_variables} and {dependent_variable}. 

    Please respond with "True" if the paper is relevant to the research topic, and "False" if it is not.
    Be generous in determining if a paper is relevant.

    Instructions:
    1. As long as the paper has both of the variables in the data, the paper is relevant.
    2. It should be considered relevant if the variables are not directly mentioned but generally related to the research topic.
    3. It's possible both variables appeared as the independent variables in the paper provided.

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


class ExtractSamplesBasicInfo(BaseModel):
    sample_info: List[SampleBasicInfo] = Field(
        description="""
        First determine how many different samples were used in the paper.
        Then, for each sample, extract the basic information about the sample.
        """
    )

EXTRACT_SAMPLES_BASIC_INFO_PROMPT = PromptTemplate.from_template(
    """
    You are a intelligent academic researcher. 
    Given a research paper, your task is to extract the sample information from the paper content.

    Instructions:
    1. Determine how many different samples were used in the paper.
    2. For each sample, first provide a name and description to the sample. \
        Thenextract the basic information about the sample.

    For each sample, please extract the following information:
    - sample_name: The name of the sample. If the authers didn't provide a name, you can generate a name based on the information you see. Please make sure different samples have different names.
    - sample_description: A description of the sample.
    - country: The country where the sample was collected
    - sampling_technique: The sampling technique used (choose from "National representative sample", "Urban/City sample", "Rural/Village sample", or "Other")
    - sample_type: Brief notes about the source of the sample (e.g., 'clinic' for clinical samples)
    - sample_size: The number of participants in the sample (Sample data size N)
    - mean_age: The average age of participants
    - sd_age: The standard deviation of the ages
    - male_n: The number of male participants
    - female_n: The number of female participants
    - major_ethnicity: The major ethnicity of the sample
    - major_ethnicity_percentage: The percentage of the major ethnicity
    - response_rate: The percentage of participants who responded to the study

    Research Paper:
    {paper_content}

    Output:
    """
)

def get_extract_samples_basic_info_chain(llm = GPT4O_LANGCHAIN_NEW):
    return EXTRACT_SAMPLES_BASIC_INFO_PROMPT | llm.with_structured_output(ExtractSamplesBasicInfo)


class ExtractVariablesInfoFromSample(BaseModel):
    variables_info: List[VariableInfoInSample] = Field(description="The desired information for each variable")
    correlations_info: List[CorrelationInfoInSample] = Field(description="The correlations between any pairs of variables")

EXTRACT_VARIABLES_INFO_FROM_SAMPLE_PROMPT = PromptTemplate.from_template(
    """
    You are a intelligent academic researcher. 
    Given a research paper, your task is to extract the variable information in a specific sample from the paper content.

    Instructions:
    1. You will be given the sample description - please complete the task by only using content related to the sample.
    2. For each of the given variables, extract the basic information about the variable related to the sample.

    For each variable, please extract the following information:
    - variable_name: The name of the variable.
    - variable_type: The type of the variable.
    - scale_measure: The scale or measure used for the variable.
    - reliability: The reliability (e.g., Cronbach's alpha) of the scale used.
    - mean: The mean value of the variable.
    - standard_deviation: The standard deviation of the variable.

    3. For each pair of variables, determine if there is information on correlation between them. If so, extract the correlation information.
    - variable_pair: The pair of variables, represented as a tuple of two strings, sorted alphabetically. Remember to extract for any pairs.
    - exists: Whether the correlation exists for a pair of variables.
    - correlation_coefficient: The correlation coefficient between a pair of variables.

    Sample Description:
    {sample_description}

    Variables: {variables}

    Paper Content:
    {paper_content}

    Output:
    """
)

def get_extract_variables_info_from_sample_chain(llm = GPT4O_LANGCHAIN_NEW):
    return EXTRACT_VARIABLES_INFO_FROM_SAMPLE_PROMPT | llm.with_structured_output(ExtractVariablesInfoFromSample)

