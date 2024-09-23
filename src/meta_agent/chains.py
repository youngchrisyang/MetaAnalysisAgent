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
    dependent_variable: str = Field(description="The dependent variable")
    independent_variables: List[str] = Field(description="The independent variables")
    dependent_variable_info: VariableInfoInSample = Field(description="The information of the dependent variable")
    independent_variables_info: List[VariableInfoInSample] = Field(description="The information of the independent variables")
    correlation_with_dependent_variable: List[CorrelationInfoInSample] = Field(description="The correlation between the independent variables and the dependent variable")

EXTRACT_VARIABLES_INFO_FROM_SAMPLE_PROMPT = PromptTemplate.from_template(
    """
    You are a intelligent academic researcher. 
    Given a research paper, your task is to extract the variable information in a specific sample from the paper content.

    Instructions:
    1. You will be given the sample description - please complete the task by only using content related to the sample.
    2. For each variable (including both dependent and independent variables), extract the basic information about the variable.

    For each variable, please extract the following information:
    - variable_name: The name of the variable.
    - variable_type: The type of the variable.
    - scale_measure: The scale or measure used for the variable.
    - reliability: The reliability (e.g., Cronbach's alpha) of the scale used.
    - mean: The mean value of the variable.
    - standard_deviation: The standard deviation of the variable.

    3. For each independent variable, determine if there is a data on correlation between the dependent variable and independent variables. If so, extract the correlation information.

    Sample Description:
    {sample_description}

    Dependent Variable: {dependent_variable}

    Independent Variables: {independent_variables}

    Paper Content:
    {paper_content}

    Output:
    """
)

def get_extract_variables_info_from_sample_chain(llm = GPT4O_LANGCHAIN_NEW):
    return EXTRACT_VARIABLES_INFO_FROM_SAMPLE_PROMPT | llm.with_structured_output(ExtractVariablesInfoFromSample)

