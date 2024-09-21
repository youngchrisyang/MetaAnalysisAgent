from pydantic import BaseModel, Field
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict, Union

class PaperMetaInfo(BaseModel):
    paper_id: str = Field(description="The ID of the paper")
    title: str = Field(description="The title of the paper")
    publication_year: int = Field(description="The year the paper was published")
    journal: str = Field(description="The name of the journal")
    published_status: Literal["published", "unpublished"]    = Field(description="The publication status of the paper")
    publication_type: Literal["journal article", "dissertation paper", "conference paper"] = Field(description="The type of publication of the paper")
    num_studies: int = Field(description="The number of different studies conducted in the paper")
    num_samples: int = Field(description="The number of different samples used in the paper")

class SampleInfo(BaseModel):
    sample_name: str = Field(description="The alias or short name of the sample")
    country: str = Field(description="The country where the sample was collected")
    sampling_technique: Literal["National representative sample", "Urban/City sample", "Rural/Village sample"] = Field(description="The sampling technique used")
    sample_type: str = Field(description="Brief notes about the source of the sample (e.g., 'clinic' for clinical samples)")
    sample_size: int = Field(description="The number of participants in the sample")
    mean_age: Optional[float] = Field(description="The average age of participants")
    sd_age: Optional[float] = Field(description="The standard deviation of the ages")
    male_n: Optional[int] = Field(description="The number of male participants")
    female_n: Optional[int] = Field(description="The number of female participants")
    major_ethnicity: Optional[str] = Field(description="The major ethnicity of the sample")
    major_ethnicity_percentage: Optional[float] = Field(description="The percentage of the major ethnicity")
    response_rate: Optional[float] = Field(description="The percentage of participants who responded to the study")

class VariableInfo(BaseModel):
    variable_name: str = Field(description="The name of the variable")
    variable_type: Literal["Continuous", "Categorical"] = Field(description="The type of the variable")
    scale_measure: str = Field(description="The scale or measure used for the variable")
    reliability: Optional[float] = Field(description="The reliability (e.g., Cronbach's alpha) of the scale used")
    mean: Optional[float] = Field(description="The mean value of the variable")
    standard_deviation: Optional[float] = Field(description="The standard deviation of the variable")

class VariablePairInfo(BaseModel):
    variable_x: VariableInfo = Field(description="The first variable in the pair")
    variable_y: VariableInfo = Field(description="The second variable in the pair")
    correlation_coefficient: float = Field(description="The correlation coefficient between Variable X and Variable Y")

class MetaAnalysisInfo(BaseModel):
    paper_meta_info: PaperMetaInfo = Field(description="The meta information of the paper")
    sample_info: List[SampleInfo] = Field(description="The sample information of the paper")
    variable_pair_info: List[VariablePairInfo] = Field(description="The variable pair information of the paper")
