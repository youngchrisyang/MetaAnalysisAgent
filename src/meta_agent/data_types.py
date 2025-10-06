from pydantic import BaseModel,Field
from typing import List, Literal, Optional, Dict

from src.utils.configs import (
    DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS,
    DEFAULT_PAPER_META_INFO_INSTRUCTIONS,
    DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS,
    DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS
)

## Define fields to extract for each of the following categories
class PaperMetaInfo(BaseModel):
    paper_id: Optional[str] = Field(description="The ID of the paper")
    title: Optional[str] = Field(description="The title of the paper")
    publication_year: Optional[int] = Field(description="The year the paper was published")
    journal: Optional[str] = Field(description="The name of the journal")
    published_status: Optional[Literal["published", "unpublished"]] = Field(description="The publication status of the paper")
    publication_type: Optional[Literal["journal article", "dissertation paper", "conference paper", "book chapter", "preprint"]] = Field(description="The type of publication of the paper")
    authors: Optional[List[str]] = Field(description="The authors of the paper")
    num_studies: Optional[int] = Field(description="The number of studies in the paper")
    study_design_types: Optional[List[Literal["cross-sectional", "longitudinal", "cross-sectional-longitudinal", "lab experiment", "field experiment", "natural experiment", "not specified"]]] = Field(description="The study design type of the paper")
    num_samples: Optional[int] = Field(description="The number of different samples used in the paper. Note this is different from sample size. ")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(description="Overall confidence in the accuracy of the extracted paper meta-information")
    reasons: List[str] = Field(description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the extracted information")

class SampleBasicInfo(BaseModel):
    sample_name: str = Field(description="The alias or short name of the sample")
    sample_description: str = Field(description="A brief description of the sample")
    country: str = Field(description="The country where the sample was collected")
    sampling_technique: Literal["National representative sample", "Urban/City sample", "Rural/Village sample", "Other"] = Field(description="The sampling technique used")
    sample_type: str = Field(description="Brief notes about the source of the sample (e.g., 'clinic' for clinical samples)")
    sample_size: int = Field(description="The number of participants in the sample. Sample data size N")
    mean_age: Optional[float] = Field(description="The average age of participants")
    sd_age: Optional[float] = Field(description="The standard deviation of the ages")
    male_n: Optional[int] = Field(description="The number of male participants explicitly reported. Do not use total sample size to calculate the number of male participants.")
    female_n: Optional[int] = Field(description="The number of female participants explicitly reported. Do not use total sample size to calculate the number of female participants.")
    major_ethnicity: Optional[str] = Field(description="The major ethnicity of the sample")
    major_ethnicity_percentage: Optional[float] = Field(description="The percentage of the major ethnicity")
    response_rate: Optional[float] = Field(description="The percentage of participants who responded to the study")
    original_sentences: Optional[str] = Field(description="For reference purpose, the original sentences from the paper that describe the sample basic information. If in a table, use the table name.")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(description="Confidence in the accuracy of the extracted sample information")
    reasons: List[str] = Field(description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the sample information")

class VariableInfoInSample(BaseModel):
    variable_name: str = Field(description="The exact name of the variable as used in the paper")
    variable_type: Literal["Continuous", "Categorical"] = Field(description="The type of the variable")
    scale_measure: str = Field(description="The scale or measure used for the variable")
    scale_range: Optional[str] = Field(description="The range of the scale used")
    reliability: Optional[float] = Field(description="The reliability (e.g., Cronbach's alpha) of the scale used")
    mean: Optional[float] = Field(description="The mean value of the variable")
    standard_deviation: Optional[float] = Field(description="The standard deviation of the variable")
    time_point: Optional[str] = Field(description="The time point of the variable")
    conceptual_category: Optional[str] = Field(description="Conceptual category (e.g., cognitive, behavioral, biological)")
    original_sentences: Optional[str] = Field(description="For reference purpose, the original sentences from the paper that describe the variable. If in a table, use the table name.")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(description="Confidence in the accuracy of the extracted variable information")
    reasons: List[str] = Field(description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the variable information")

# New model for Step 1: Identifying all related variables
class IdentifyRelatedVariables(BaseModel):
    related_variables: List[VariableInfoInSample] = Field(
        description="All variables in this sample that are related to the specified dependent and independent variables, including variants and different names used in the paper"
    )

# New model for Step 2: Variable pair for correlation extraction
class VariablePair(BaseModel):
    variable1_name: str = Field(description="Name of the first variable in the pair")
    variable2_name: str = Field(description="Name of the second variable in the pair")

class CorrelationInfoInSample(BaseModel):
    variable1: str = Field(description="Name of the first variable in the pair")
    variable2: str = Field(description="Name of the second variable in the pair")
    correlation_type: Optional[Literal["Pearson", "Spearman", "Kendall", "Not specified"]] = Field(description="The type of the correlation")
    correlation_coefficient: Optional[float] = Field(None, description="The correlation coefficient between the pair of variables")
    correlation_p_value: Optional[float] = Field(None, description="P-value or significance level if reported")
    correlation_confidence_interval_lower: Optional[float] = Field(None, description="Lower bound of 95% confidence interval if reported")
    correlation_confidence_interval_upper: Optional[float] = Field(None, description="Upper bound of 95% confidence interval if reported")
    adjusted_or_zero_order: Optional[Literal["zero-order","adjusted/partial","not specified"]] = Field(description="Zero or adjusted correlation")
    significance_level: Optional[str] = Field(None, description="The significance level of the correlation")
    original_sentences: Optional[str] = Field(None, description="For reference purpose, the original sentences from the paper that describe the correlation. If in a table, use the table name.")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(None, description="Confidence in the accuracy of the extracted correlation information")
    reasons: List[str] = Field(None, description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the correlation information")

# New model for individual group information in between-group comparisons
class GroupInfo(BaseModel):
    group_name: str = Field(description="Name or label of the group/condition")
    group_description: Optional[str] = Field(description="Description of the group/condition if provided")
    group_mean: Optional[float] = Field(description="Mean value for this group")
    group_sd: Optional[float] = Field(description="Standard deviation for this group")
    group_n: Optional[int] = Field(description="Sample size for this group")
    group_median: Optional[float] = Field(description="Median value for this group if reported")
    group_min: Optional[float] = Field(description="Minimum value for this group if reported")
    group_max: Optional[float] = Field(description="Maximum value for this group if reported")

# New model for Between-Group Effects (III-2) - Updated for multiple groups
class BetweenGroupEffectInSample(BaseModel):
    outcome_variable: str = Field(description="Name of the outcome variable being compared across groups")
    grouping_variable: str = Field(description="Name of the variable that defines the groups (e.g., 'treatment condition', 'age group')")
    groups: List[GroupInfo] = Field(description="List of all groups involved in the comparison, with their respective statistics")
    exists: bool = Field(description="Whether a between-group difference exists for these variables")
    effect_type: Literal["t-test", "ANOVA", "Mann-Whitney", "Kruskal-Wallis", "Chi-square", "Not specified"] = Field(description="The type of statistical test used")
    effect_size: Optional[float] = Field(description="Effect size (e.g., Cohen's d, eta-squared, partial eta-squared) if reported")
    effect_size_type: Optional[str] = Field(description="Type of effect size measure used (e.g., 'Cohen\\'s d', 'eta-squared', 'partial eta-squared')")
    test_statistic: Optional[float] = Field(description="Test statistic value (t, F, H, chi-square, etc.)")
    degrees_of_freedom: Optional[str] = Field(description="Degrees of freedom (e.g., '2, 97' for F-test, '1' for t-test)")
    significance_level: Optional[float] = Field(description="P-value or significance level if reported")
    post_hoc_tests: Optional[str] = Field(description="Description of post-hoc tests conducted if applicable (e.g., 'Tukey HSD', 'Bonferroni')")
    original_sentences: Optional[str] = Field(description="For reference purpose, the original sentences from the paper that describe the between-group effect.")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(description="Confidence in the accuracy of the extracted between-group effect information")
    reasons: List[str] = Field(description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the between-group effect information")

# New model for Within-Subject Effects (III-3)
class WithinSubjectEffectInSample(BaseModel):
    variable_name: str = Field(description="Name of the variable being measured across conditions/time")
    condition1: str = Field(description="Name or description of the first condition/time point")
    condition2: str = Field(description="Name or description of the second condition/time point")
    exists: bool = Field(description="Whether a within-subject difference exists for this variable")
    effect_type: Literal["paired t-test", "repeated measures ANOVA", "Wilcoxon", "Mixed-effects", "Not specified"] = Field(description="The type of statistical test used")
    effect_size: Optional[float] = Field(description="Effect size (e.g., Cohen's d, partial eta-squared) if reported")
    effect_size_type: Optional[str] = Field(description="Type of effect size measure used")
    condition1_mean: Optional[float] = Field(description="Mean at the first condition/time point")
    condition2_mean: Optional[float] = Field(description="Mean at the second condition/time point")
    condition1_sd: Optional[float] = Field(description="Standard deviation at the first condition/time point")
    condition2_sd: Optional[float] = Field(description="Standard deviation at the second condition/time point")
    correlation_between_conditions: Optional[float] = Field(description="Correlation between conditions if reported")
    sample_size: Optional[int] = Field(description="Sample size for the within-subject comparison")
    test_statistic: Optional[float] = Field(description="Test statistic value (t, F, etc.)")
    significance_level: Optional[float] = Field(description="P-value or significance level if reported")
    original_sentences: Optional[str] = Field(description="For reference purpose, the original sentences from the paper that describe the within-subject effect.")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(description="Confidence in the accuracy of the extracted within-subject effect information")
    reasons: List[str] = Field(description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the within-subject effect information")

# New model for Binary Event Effects (III-4)
class BinaryEventEffectInSample(BaseModel):
    outcome_variable: str = Field(description="Name of the binary outcome variable")
    predictor_variable: str = Field(description="Name of the predictor variable")
    exists: bool = Field(description="Whether a binary event effect exists for this variable pair")
    effect_type: Literal["Odds Ratio", "Risk Ratio", "Hazard Ratio", "Logistic Regression", "Chi-square", "Not specified"] = Field(description="The type of analysis used")
    odds_ratio: Optional[float] = Field(description="Odds ratio if reported")
    risk_ratio: Optional[float] = Field(description="Risk ratio if reported")
    hazard_ratio: Optional[float] = Field(description="Hazard ratio if reported")
    confidence_interval_lower: Optional[float] = Field(description="Lower bound of 95% confidence interval")
    confidence_interval_upper: Optional[float] = Field(description="Upper bound of 95% confidence interval")
    event_count_group1: Optional[int] = Field(description="Number of events in first group")
    total_count_group1: Optional[int] = Field(description="Total count in first group")
    event_count_group2: Optional[int] = Field(description="Number of events in second group")
    total_count_group2: Optional[int] = Field(description="Total count in second group")
    test_statistic: Optional[float] = Field(description="Test statistic value (chi-square, z, etc.)")
    significance_level: Optional[float] = Field(description="P-value or significance level if reported")
    original_sentences: Optional[str] = Field(description="For reference purpose, the original sentences from the paper that describe the binary event effect.")
    confidence_level: Literal["confident", "with reservations", "not confident"] = Field(description="Confidence in the accuracy of the extracted binary event effect information")
    reasons: List[str] = Field(description="List of specific reasons explaining the confidence level and what researchers should pay attention to regarding the binary event effect information")

# New model for Step 2: Extracting correlations for variable pairs
class ExtractCorrelationsForPairs(BaseModel):
    correlations: Optional[List[CorrelationInfoInSample]] = Field(
        description="Correlation information for each pair of variables. Extract all correlations that exist in the text chunk. If no correlation is found, do not create an entry for it."
    )

# New models for the other extraction types
class ExtractBetweenGroupEffects(BaseModel):
    has_group_separations: bool = Field(
        description="Whether the study contains distinct groups or conditions that can be compared (e.g., treatment vs control, high vs low, experimental conditions). Return False for purely correlational studies without group separations."
    )
    between_group_effects: List[BetweenGroupEffectInSample] = Field(
        description="Between-group effect information for relevant variable comparisons. Leave empty if has_group_separations is False."
    )

class ExtractWithinSubjectEffects(BaseModel):
    within_subject_effects: List[WithinSubjectEffectInSample] = Field(
        description="Within-subject effect information for relevant variable comparisons across conditions or time"
    )

class ExtractBinaryEventEffects(BaseModel):
    binary_event_effects: List[BinaryEventEffectInSample] = Field(
        description="Binary event effect information for relevant variable relationships involving binary outcomes"
    )

class SampleCompleteInfo(BaseModel):
    sample_name: str = Field(description="The name of the sample")
    sample_basic_info: SampleBasicInfo = Field(description="The basic information of the sample")
    variables_info: Optional[List[VariableInfoInSample]] = Field(description="The information of all related variables found in this sample")
    correlations_info: Optional[List[CorrelationInfoInSample]] = Field(description="The correlations between any pairs of variables and any type")
    between_group_effects_info: Optional[List[BetweenGroupEffectInSample]] = Field(description="The between-group effects for relevant comparisons", default=[])
    within_subject_effects_info: Optional[List[WithinSubjectEffectInSample]] = Field(description="The within-subject effects for relevant comparisons", default=[])
    binary_event_effects_info: Optional[List[BinaryEventEffectInSample]] = Field(description="The binary event effects for relevant comparisons", default=[])

class FinalMetaAnalysisInfo(BaseModel):
    paper_meta_info: PaperMetaInfo = Field(description="The meta information of the paper")
    sample_info: List[SampleCompleteInfo] = Field(description="List of all samples in the paper")


class UserInstructions(BaseModel):
    paper_relevance_instructions: Optional[str] = Field(default=DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS)
    paper_meta_info_instructions: Optional[str] = Field(default=DEFAULT_PAPER_META_INFO_INSTRUCTIONS)
    samples_extraction_instructions: Optional[str] = Field(default=DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS)
    variables_extraction_instructions: Optional[str] = Field(default=DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS)
