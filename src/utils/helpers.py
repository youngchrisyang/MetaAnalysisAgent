import os, json
from dotenv import load_dotenv, find_dotenv
from llama_parse import LlamaParse
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.meta_agent.data_types import SampleCompleteInfo
from typing import Dict, Any

def get_llamaparsed_doc(file_path, parsing_mode="standard"):
    # Configure LlamaParse based on selected parsing mode
    if parsing_mode == "premium":
        # Enhanced LlamaParse configuration with premium features for complex academic papers
        parser = LlamaParse(
            # Core settings
            result_type="markdown",
            language='en',

            # Premium features for better accuracy
            premium_mode=True,  # Enable premium parsing for complex documents

            # Custom parsing instructions for academic papers
            parsing_instruction="""
            Extract all content from this academic paper with the highest accuracy.
            Pay special attention to:
            - Tables with statistical data (correlations, means, standard deviations)
            - Sample descriptions and demographics
            - Variable definitions and measurements
            - Research methodology sections
            - Results and statistical findings
            - Convert any diagrams to structured text descriptions
            - Preserve mathematical equations and statistical formulas
            - Maintain the logical flow and structure of the document
            """,

            # Advanced parsing options for better content extraction
            include_metadata=True,
            disable_ocr=False,  # Keep OCR enabled for scanned documents
            skip_diagonal_text=True,  # Avoid parsing watermarks/diagonal text
            preserve_very_small_text=False,  # Skip footnote symbols that might be noise

            # Table and layout options
            output_tables_as_HTML=True,  # Better table structure preservation
            merge_tables_across_pages_in_markdown=True,  # Handle tables spanning pages
            preserve_layout_alignment_across_pages=True,  # Maintain formatting

            # Image and visual content
            disable_image_extraction=False,  # Keep images for potential figure captions
            take_screenshot=False,  # Not needed for text extraction

            # Header/footer handling for academic papers
            hide_headers=True,  # Remove page headers that might interfere
            hide_footers=True,  # Remove page footers that might interfere

            # Performance and caching
            invalidate_cache=True,  # Ensure fresh parsing for accurate results
            do_not_cache=False,  # Allow caching for repeated use

            # Formatting instructions
            is_formatting_instruction=True,  # Enable custom formatting instructions
        )
    else:
        # Standard LlamaParse configuration - cost-effective and fast
        parser = LlamaParse(
            # Core settings
            result_type="markdown",
            language='en',

            # Standard features - no premium mode
            premium_mode=False,  # Use standard parsing

            # Basic metadata extraction
            include_metadata=True,

            # Basic parsing options
            disable_ocr=False,  # Keep OCR enabled for scanned documents
            skip_diagonal_text=True,  # Avoid parsing watermarks/diagonal text

            # Standard table handling
            output_tables_as_HTML=False,  # Use markdown tables
            merge_tables_across_pages_in_markdown=True,  # Handle tables spanning pages

            # Basic image handling
            disable_image_extraction=True,  # Skip images in standard mode
            take_screenshot=False,  # Not needed

            # Performance and caching
            do_not_cache=False,  # Allow caching for repeated use
        )

    # Parse the document
    document = parser.load_data(file_path)

    # Convert to langchain format while preserving metadata
    document = [doc.to_langchain_format() for doc in document]

    # Ensure page numbers are accessible in the metadata
    for doc in document:
        if 'page' not in doc.metadata and 'page_number' in doc.metadata:
            doc.metadata['page'] = doc.metadata['page_number']

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
        print(f"  {corr.variable1} - {corr.variable2}:")
        if corr.correlation_coefficient is not None:
            print(f"    Correlation Coefficient: {corr.correlation_coefficient:.2f}")
        print(f"    Type: {corr.correlation_type}")
        print()  # Add a blank line between correlations

if __name__ == "__main__":
    # Example usage:
    sample_info = SampleCompleteInfo(...)  # Your SampleCompleteInfo object
    pretty_print_sample_info(sample_info)
    