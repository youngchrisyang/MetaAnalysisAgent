from pydantic import BaseModel,Field
from typing import List, Literal, Optional, Dict

class PaperMetaInfo(BaseModel):
    paper_id: str = Field(description="The ID of the paper")
    title: str = Field(description="The title of the paper")
    publication_year: int = Field(description="The year the paper was published")
    journal: str = Field(description="The name of the journal")
    published_status: Literal["published", "unpublished"] = Field(description="The publication status of the paper")
    publication_type: Literal["journal article", "dissertation paper", "conference paper"] = Field(description="The type of publication of the paper")

class SampleBasicInfo(BaseModel):
    sample_name: str = Field(description="The alias or short name of the sample")
    sample_description: str = Field(description="A brief description of the sample")
    country: str = Field(description="The country where the sample was collected")
    sampling_technique: Literal["National representative sample", "Urban/City sample", "Rural/Village sample", "Other"] = Field(description="The sampling technique used")
    sample_type: str = Field(description="Brief notes about the source of the sample (e.g., 'clinic' for clinical samples)")
    sample_size: int = Field(description="The number of participants in the sample. Sample data size N")
    mean_age: Optional[float] = Field(description="The average age of participants")
    sd_age: Optional[float] = Field(description="The standard deviation of the ages")
    male_n: Optional[int] = Field(description="The number of male participants")
    female_n: Optional[int] = Field(description="The number of female participants")
    major_ethnicity: Optional[str] = Field(description="The major ethnicity of the sample")
    major_ethnicity_percentage: Optional[float] = Field(description="The percentage of the major ethnicity")
    response_rate: Optional[float] = Field(description="The percentage of participants who responded to the study")

class VariableInfoInSample(BaseModel):
    variable_name: str = Field(description="The name of the variable")
    variable_type: Literal["Continuous", "Categorical"] = Field(description="The type of the variable")
    scale_measure: str = Field(description="The scale or measure used for the variable")
    reliability: Optional[float] = Field(description="The reliability (e.g., Cronbach's alpha) of the scale used")
    mean: Optional[float] = Field(description="The mean value of the variable")
    standard_deviation: Optional[float] = Field(description="The standard deviation of the variable")

class CorrelationInfoInSample(BaseModel):
    variable1: str = Field(description="The first variable in the pair")
    variable2: str = Field(description="The second variable in the pair")
    exists: bool = Field(description="Whether the correlation exists for an independent variable and the dependent variable")
    correlation_coefficient: Optional[float] = Field(description="The correlation coefficient between a pair of variables")

class SampleCompleteInfo(BaseModel):
    sample_name: str = Field(description="The name of the sample")
    sample_basic_info: SampleBasicInfo = Field(description="The basic information of the sample")
    variables_info: List[VariableInfoInSample] = Field(description="The information of the variables")
    correlations_info: List[CorrelationInfoInSample] = Field(description="The correlations between any pairs of variables")

class FinalMetaAnalysisInfo(BaseModel):
    paper_meta_info: PaperMetaInfo = Field(description="The meta information of the paper")
    sample_info: List[SampleCompleteInfo] = Field(description="List of all samples in the paper")
