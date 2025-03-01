import os, json
from dotenv import load_dotenv, find_dotenv
from llama_parse import LlamaParse
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.meta_agent.data_types import SampleCompleteInfo
from typing import Dict, Any

def get_llamaparsed_doc(file_path):
    document = LlamaParse(result_type="markdown", language='en').load_data(file_path)
    document = [doc.to_langchain_format() for doc in document]
    return document
    
def pretty_print_sample_info(sample_info: SampleCompleteInfo):
    def format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def print_section(title: str, data: Dict[str, Any]):
        print(f"\n{title}:")
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            formatted_value = format_value(value)
            print(f"  {formatted_key}: {formatted_value}")

    print(f"Sample Name: {sample_info.sample_name}")
    
    print_section("Basic Information", sample_info.sample_basic_info.dict())
    
    print("\nVariables Information:")
    for var in sample_info.variables_info:
        print(f"  {var.variable_name}:")
        var_dict = var.dict()
        for key, value in var_dict.items():
            if key != 'variable_name':
                formatted_key = key.replace('_', ' ').title()
                formatted_value = format_value(value)
                print(f"    {formatted_key}: {formatted_value}")
        print()  # Add a blank line between variables

    print("Correlations Information:")
    for corr in sample_info.correlations_info:
        var1, var2 = corr.variable_pair
        print(f"  {var1} - {var2}:")
        print(f"    Exists: {corr.exists}")
        if corr.correlation_coefficient is not None:
            print(f"    Correlation Coefficient: {corr.correlation_coefficient:.2f}")
        print()  # Add a blank line between correlations

if __name__ == "__main__":
    # Example usage:
    sample_info = SampleCompleteInfo(...)  # Your SampleCompleteInfo object
    pretty_print_sample_info(sample_info)
    