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
from src.utils.configs import CONFIDENCE_LEVEL_INSTRUCTIONS, WARNINGS_CALLOUTS_INSTRUCTIONS
from .data_types import (
    PaperMetaInfo, 
    SampleBasicInfo, 
    VariableInfoInSample, 
    CorrelationInfoInSample,
    IdentifyRelatedVariables,
    ExtractCorrelationsForPairs,
    ExtractBetweenGroupEffects,
    ExtractWithinSubjectEffects,
    ExtractBinaryEventEffects,
    GroupInfo,
    VariablePair
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
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
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
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
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

# ========== NEW CHAIN 1: IDENTIFY ALL RELATED VARIABLES ==========
IDENTIFY_RELATED_VARIABLES_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to identify ALL variables in a specific sample that are related to the research topic."),
    ("human", """
    I need you to identify ALL variables **in this specific sample** that are related to our research variables.
    
    Research Focus:
    - Dependent Variable: {dependent_variable}
    - Independent Variables: {independent_variables}
    
    Sample Information:
    {sample_description}
    
    Your task:
    1. **Find ALL variables in this sample that are variants, sub measures, or same construct but different names to the dependent and independent variables listed above**
    2. Include the EXACT names as they appear in the paper (not the generic names I provided)
    3. Include sub-scales, composite scores, different measurement methods, and time points
    4. Extract measurement details, descriptive statistics, and reliability information when available
    
    For each variable, please extract the following information:
    {fields_to_extract}
    
    Paper Content:
    {paper_content}
    """),
    ("human", """
    Please follow these additional instructions when identifying related variables:
    {instructions}
    """),
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
    """),
])

def get_identify_related_variables_chain(llm = GPT41_LANGCHAIN):
    # Generate the fields to extract based on the VariableInfoInSample model
    fields = []
    for field_name in VariableInfoInSample.model_fields:
        field_obj = VariableInfoInSample.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = IDENTIFY_RELATED_VARIABLES_PROMPT | llm.with_structured_output(IdentifyRelatedVariables)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain

# ========== NEW CHAIN 2: EXTRACT CORRELATIONS FOR VARIABLE PAIRS ==========
EXTRACT_CORRELATIONS_FOR_PAIRS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to extract correlation information between specific pairs of variables from research papers."),
    ("human", """
    I need you to extract correlation information for the following specific pairs of variables in this sample.
    
    Sample Information:
    {sample_description}
    
    Variable Pairs to Check:
    {variable_pairs}
    
    Your task:
    For each pair of variables listed above, determine:
    1. Whether a correlation between these two variables exists in the paper for this sample
    2. If it exists, extract the correlation coefficient, type (Pearson/Spearman/Kendall), and significance level
    3. Find the exact sentences or table where this information appears
    
    Look for correlations in:
    - Correlation matrices or tables
    - Results sections describing relationships
    - Statistical analyses mentioning these variable pairs
    - Any numerical relationships between the specified variables
    
    For each pair, please extract the following information:
    {fields_to_extract}
    
    Paper Content:
    {paper_content}
    """),
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
    """),
])

def get_extract_correlations_for_pairs_chain(llm = GPT41_LANGCHAIN):
    # Generate the fields to extract based on the CorrelationInfoInSample model
    fields = []
    for field_name in CorrelationInfoInSample.model_fields:
        field_obj = CorrelationInfoInSample.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_CORRELATIONS_FOR_PAIRS_PROMPT | llm.with_structured_output(ExtractCorrelationsForPairs)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain

# ========== NEW CHAIN 3: EXTRACT BETWEEN-GROUP EFFECTS ==========
EXTRACT_BETWEEN_GROUP_EFFECTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to first determine if a study has group separations, then extract between-group effects information if they exist and have sufficient statistical data."),
    ("human", """
    I need you to analyze this sample and determine if it contains group separations, then extract between-group effects if applicable.
    
    Sample Information:
    {sample_description}
    
    Variables Available:
    {variables_list}
    
    **STEP 1: Determine if Group Separations Exist**
    First, carefully assess whether this study involves distinct groups or conditions that can be compared. Look for:
    
    INDICATORS OF GROUP SEPARATIONS:
    - Different treatment conditions (treatment vs control, placebo vs active)
    - Demographic groups (male vs female, different age groups, high vs low income)
    - Experimental conditions (different tasks, environments, or manipulations)  
    - Categories/levels of a factor (high vs medium vs low performance, different job types)
    - Intervention vs comparison groups
    - Before/after with separate control groups
    - Different cohorts or participant types
    
    STATISTICAL EVIDENCE:
    - Independent samples t-tests, ANOVA, Mann-Whitney U, Kruskal-Wallis
    - Chi-square tests for group differences
    - Mentions of "between groups", "group differences", "group comparisons"
    
    NOT GROUP SEPARATIONS (purely correlational):
    - Only correlation analyses between continuous variables
    - Regression analyses without group comparisons
    - Single-group studies measuring relationships between variables
    - Studies that only report correlations, not group differences
    
    **STEP 2: Validate Statistical Data Availability**
    If group separations exist, check if sufficient statistical data is available for each group:
    
    REQUIRED GROUP DATA (for each group in the comparison):
    - Group means (most critical)
    - Group standard deviations (most critical)
    - Group sample sizes (most critical)
    - Group names/descriptions
    
    DATA SUFFICIENCY CRITERIA:
    - Each group must have at least 2 out of 3 key statistical fields: mean, SD, or N
    - At least 70% of groups in a comparison must have sufficient statistical data
    - If groups are mentioned conceptually but lack statistical breakdowns, do not extract
    
    **STEP 3: Extract Between-Group Effects (Only if Groups Exist AND Have Data)**
    Only extract effects where:
    1. Distinct groups exist in the study design
    2. Statistical comparisons between groups are reported
    3. Sufficient group-level statistical data is available (means, SDs, Ns)
    
    Look for statistical comparisons between groups:
    - Independent samples t-tests comparing two groups
    - One-way ANOVA comparing multiple groups (3+)
    - Two-way or factorial ANOVA with multiple factors
    - Mann-Whitney U tests for non-parametric two-group comparisons
    - Kruskal-Wallis tests for non-parametric multiple-group comparisons
    - Chi-square tests for categorical group differences
    - Any other between-group statistical comparisons
    
    Focus on:
    1. Identifying the grouping variable that defines the groups (e.g., "treatment condition", "age group", "gender")
    2. Identifying the outcome variable being compared across groups
    3. Extracting information for ALL groups involved (not just two):
       - Group names/labels and descriptions
       - Means and standard deviations for each group (REQUIRED)
       - Sample sizes for each group (REQUIRED)
       - Any additional descriptive statistics
    4. Recording overall test statistics, effect sizes, and significance levels
    5. Noting any post-hoc tests conducted for multiple comparisons
    
    **Important Guidelines**: 
    - Set has_group_separations = True ONLY if distinct groups exist AND have statistical data
    - Set has_group_separations = False for purely correlational studies OR studies with groups but no statistical breakdowns
    - If has_group_separations = False, leave between_group_effects empty
    - If has_group_separations = True, only include effects where groups have sufficient statistical data
    - Do NOT extract effects where groups are mentioned conceptually but lack means, SDs, or sample sizes
    - Prioritize data quality over quantity - better to have fewer high-quality extractions
    
    For each between-group comparison (if groups exist AND have data), please extract the following information:
    {fields_to_extract}
    
    Paper Content:
    {paper_content}
    
    Please provide your analysis in the specified format, ensuring all extracted effects have sufficient group-level statistical data.
    """)])

def get_extract_between_group_effects_chain(llm = GPT41_LANGCHAIN):
    from .data_types import BetweenGroupEffectInSample
    
    # Generate the fields to extract based on the BetweenGroupEffectInSample model
    fields = []
    for field_name in BetweenGroupEffectInSample.model_fields:
        field_obj = BetweenGroupEffectInSample.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_BETWEEN_GROUP_EFFECTS_PROMPT | llm.with_structured_output(ExtractBetweenGroupEffects)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain

# ========== NEW CHAIN 4: EXTRACT WITHIN-SUBJECT EFFECTS ==========
EXTRACT_WITHIN_SUBJECT_EFFECTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to extract within-subject effects information from research papers."),
    ("human", """
    I need you to extract within-subject effects information from this sample.
    
    Sample Information:
    {sample_description}
    
    Variables Available:
    {variables_list}
    
    Your task:
    Look for statistical comparisons within the same participants across different conditions or time points. This includes:
    - Paired t-tests comparing before/after measurements
    - Repeated measures ANOVA for multiple time points or conditions
    - Wilcoxon signed-rank tests for non-parametric repeated measures
    - Mixed-effects models with within-subject factors
    - Longitudinal analyses comparing the same variables over time
    
    Focus on:
    1. Identifying variables measured at multiple time points or conditions
    2. Extracting means and standard deviations for each condition/time point
    3. Recording correlations between repeated measures if available
    4. Capturing test statistics, effect sizes, and significance levels
    
    For each within-subject comparison, please extract the following information:
    {fields_to_extract}
    
    Paper Content:
    {paper_content}
    """),
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
    """),
])

def get_extract_within_subject_effects_chain(llm = GPT41_LANGCHAIN):
    from .data_types import WithinSubjectEffectInSample
    
    # Generate the fields to extract based on the WithinSubjectEffectInSample model
    fields = []
    for field_name in WithinSubjectEffectInSample.model_fields:
        field_obj = WithinSubjectEffectInSample.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_WITHIN_SUBJECT_EFFECTS_PROMPT | llm.with_structured_output(ExtractWithinSubjectEffects)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain

# ========== NEW CHAIN 5: EXTRACT BINARY EVENT EFFECTS ==========
EXTRACT_BINARY_EVENT_EFFECTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent academic researcher. Your task is to extract binary event and odds ratio information from research papers."),
    ("human", """
    I need you to extract binary event effects information from this sample.
    
    Sample Information:
    {sample_description}
    
    Variables Available:
    {variables_list}
    
    Your task:
    Look for analyses involving binary outcomes (yes/no, success/failure, presence/absence) and their relationships with predictor variables. This includes:
    - Odds ratios from logistic regression
    - Risk ratios or relative risks
    - Hazard ratios from survival analysis
    - Chi-square tests with binary outcomes
    - Count data and frequency analyses
    - Binary classification results
    
    Focus on:
    1. Identifying binary outcome variables (dichotomous dependent variables)
    2. Identifying predictor variables associated with these outcomes
    3. Extracting odds ratios, risk ratios, hazard ratios with confidence intervals
    4. Recording event counts and total counts for each group
    5. Capturing test statistics and significance levels
    
    For each binary event analysis, please extract the following information:
    {fields_to_extract}
    
    Paper Content:
    {paper_content}
    """),
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
    """),
])

def get_extract_binary_event_effects_chain(llm = GPT41_LANGCHAIN):
    from .data_types import BinaryEventEffectInSample
    
    # Generate the fields to extract based on the BinaryEventEffectInSample model
    fields = []
    for field_name in BinaryEventEffectInSample.model_fields:
        field_obj = BinaryEventEffectInSample.model_fields[field_name]
        description = field_obj.description
        fields.append(f"- {field_name}: {description}")
    
    fields_to_extract = "\n".join(fields)
    
    # Create the chain with the dynamically generated fields
    chain = EXTRACT_BINARY_EVENT_EFFECTS_PROMPT | llm.with_structured_output(ExtractBinaryEventEffects)
    
    # Return a function that will inject the fields_to_extract into the prompt
    def invoke_chain(inputs):
        inputs["fields_to_extract"] = fields_to_extract
        return chain.invoke(inputs)
    
    return invoke_chain

# ========== HTML REPORT GENERATION CHAIN ==========
HTML_REPORT_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an experienced academic researcher and technical writer. Your task is to create a comprehensive, professional HTML report that synthesizes meta-analysis extraction results for researchers.

Your report should be:
- Clear and professional
- Research-focused with practical guidance
- Concise but comprehensive
- Well-formatted with tables and bullet points
- Focused on quality control and limitations"""),
    ("human", """
Please create a comprehensive HTML report based on the following extracted meta-analysis data:

**Research Focus:**
- Dependent Variable: {dependent_variable}
- Independent Variables: {independent_variables}
- Paper Relevance: {paper_relevance}

**Paper Information:**
{paper_meta_info}

**Samples Information:**
{samples_info}

**Variables Information:**
{variables_info}

**Correlations Information:**
{correlations_info}

**Instructions for Report Generation:**

1. **Report Structure:** Create a professional HTML document with:
   - Modern CSS styling (similar to academic reports)
   - Clear sections with headers
   - Summary statistics at the top
   - Detailed findings organized by sample

2. **Content Focus:**
   - Synthesize the findings into a coherent narrative
   - Use tables for structured data (sample characteristics, variables, correlations)
   - Use bullet points for key findings and recommendations
   - Highlight confidence levels and quality concerns
   - Provide specific guidance on where researchers should intervene

3. **Quality Control Emphasis:**
   - Create a "Quality Assessment" section highlighting confidence levels
   - Summarize all reasons and their implications
   - Provide actionable recommendations for researchers
   - Identify data gaps and reliability concerns

4. **Research Guidance:**
   - Suggest areas needing manual verification
   - Recommend additional data collection where needed
   - Highlight methodological considerations
   - Provide guidance on using these results in meta-analysis

5. **Formatting Requirements:**
   - Use professional color scheme (blues, grays, whites)
   - Make tables responsive and well-formatted
   - Use icons or symbols to highlight important information
   - Include confidence indicators (colors/badges)
   - Keep the report concise but informative

Generate a complete HTML document (without the ```html``` tags) that researchers can use to understand their extracted data and make informed decisions about data quality and next steps.
"""),
])

def get_generate_html_report_chain(llm = GPT4O_LANGCHAIN_NEW):
    """Generate HTML report using LLM"""
    chain = HTML_REPORT_GENERATION_PROMPT | llm | StrOutputParser()
    return chain

# ========== LEGACY CHAIN (kept for compatibility) ==========
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
    ("human", f"""
    {CONFIDENCE_LEVEL_INSTRUCTIONS}
    
    {WARNINGS_CALLOUTS_INSTRUCTIONS}
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
