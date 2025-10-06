import os
import csv
import logging
from datetime import datetime
from typing import List, Dict
from src.meta_agent.graph import MetaAnalysisGraph
from src.meta_agent.data_types import FinalMetaAnalysisInfo, SampleCompleteInfo, PaperMetaInfo, VariableInfoInSample, CorrelationInfoInSample, SampleBasicInfo, BetweenGroupEffectInSample, WithinSubjectEffectInSample, BinaryEventEffectInSample, GroupInfo
from src.utils.initialization import initialize_env

initialize_env()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler to make logs visible
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def format_list_for_csv(field_value):
    """
    Convert list fields (like reasons) to comma-separated strings for CSV compatibility.
    """
    if isinstance(field_value, list):
        if field_value:  # If list is not empty
            return "; ".join([str(item) for item in field_value])
        else:  # If list is empty
            return ""
    return field_value

def sanitize_filename(filename):
    """
    Sanitize a string to be safe for use as a filename by removing/replacing invalid characters.
    """
    import re
    # Remove or replace characters that are invalid in filenames
    # Replace slashes, backslashes, colons, etc. with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', str(filename))
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_{2,}', '_', sanitized)
    # Remove leading/trailing underscores and spaces
    sanitized = sanitized.strip('_ ')
    # Ensure the filename isn't too long (max 200 chars for safety)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

def process_papers(input_dir: str, output_dir: str, dependent_variable: str, independent_variables: List[str], user_instructions=None, effect_types_to_extract: List[str] = None, target_groups_for_comparison: str = None, selected_model: str = None, parsing_mode: str = None, progress_callback=None):
    # Set default model if none provided
    if selected_model is None:
        selected_model = "LLM_GPT41_FALLBACK"

    # Set default parsing mode if none provided
    if parsing_mode is None:
        parsing_mode = "standard"

    meta_analysis = MetaAnalysisGraph(selected_model=selected_model, parsing_mode=parsing_mode)
    graph = meta_analysis.create_meta_analysis_graph()

    # Set default values for effect extraction configuration
    if effect_types_to_extract is None:
        effect_types_to_extract = ["corr_r"]  # Default to correlation analysis only

    # Log the selected effect types for debugging
    logging.info(f"Effect types to extract: {effect_types_to_extract}")

    # Prepare output CSV files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)

    # Always create core files
    papers_file = open(os.path.join(timestamped_output_dir, 'papers.csv'), 'w', newline='', encoding='utf-8')
    samples_file = open(os.path.join(timestamped_output_dir, 'samples.csv'), 'w', newline='', encoding='utf-8')
    variables_file = open(os.path.join(timestamped_output_dir, 'variables.csv'), 'w', newline='', encoding='utf-8')

    papers_writer = csv.writer(papers_file)
    samples_writer = csv.writer(samples_file)
    variables_writer = csv.writer(variables_file)

    # Conditionally create effect-specific CSV files based on selected effect types
    correlations_file = None
    correlations_writer = None
    if "corr_r" in effect_types_to_extract:
        correlations_file = open(os.path.join(timestamped_output_dir, 'correlations.csv'), 'w', newline='', encoding='utf-8')
        correlations_writer = csv.writer(correlations_file)

    between_group_effects_file = None
    between_group_effects_writer = None
    if "indep_d" in effect_types_to_extract:
        between_group_effects_file = open(os.path.join(timestamped_output_dir, 'between_group_effects.csv'), 'w', newline='', encoding='utf-8')
        between_group_effects_writer = csv.writer(between_group_effects_file)

    within_subject_effects_file = None
    within_subject_effects_writer = None
    if "paired_d" in effect_types_to_extract:
        within_subject_effects_file = open(os.path.join(timestamped_output_dir, 'within_subject_effects.csv'), 'w', newline='', encoding='utf-8')
        within_subject_effects_writer = csv.writer(within_subject_effects_file)

    binary_event_effects_file = None
    binary_event_effects_writer = None
    if "binary_or" in effect_types_to_extract:
        binary_event_effects_file = open(os.path.join(timestamped_output_dir, 'binary_event_effects.csv'), 'w', newline='', encoding='utf-8')
        binary_event_effects_writer = csv.writer(binary_event_effects_file)

    # Write headers dynamically based on the model fields
    papers_writer.writerow(['paper_id', 'title', 'publication_year', 'journal', 'published_status', 'publication_type', 'authors', 'is_relevant', 'confidence_level', 'reasons'])

    # Get sample fields dynamically from the model
    sample_fields = ['paper_id', 'sample_id', 'sample_name'] + list(SampleBasicInfo.__annotations__.keys())
    samples_writer.writerow(sample_fields)

    # Get variable fields dynamically
    variable_fields = ['paper_id', 'sample_id'] + list(VariableInfoInSample.__annotations__.keys())
    variables_writer.writerow(variable_fields)

    # Conditionally write headers for effect-specific files
    if correlations_writer:
        correlation_fields = ['paper_id', 'sample_id'] + list(CorrelationInfoInSample.__annotations__.keys())
        correlations_writer.writerow(correlation_fields)

    if between_group_effects_writer:
        # Get between-group effects fields dynamically (flattened structure)
        between_group_base_fields = [field for field in BetweenGroupEffectInSample.__annotations__.keys() if field != 'groups']
        group_fields = list(GroupInfo.__annotations__.keys())
        between_group_effects_fields = ['paper_id', 'sample_id'] + between_group_base_fields + group_fields
        between_group_effects_writer.writerow(between_group_effects_fields)

    if within_subject_effects_writer:
        within_subject_effects_fields = ['paper_id', 'sample_id'] + list(WithinSubjectEffectInSample.__annotations__.keys())
        within_subject_effects_writer.writerow(within_subject_effects_fields)

    if binary_event_effects_writer:
        binary_event_effects_fields = ['paper_id', 'sample_id'] + list(BinaryEventEffectInSample.__annotations__.keys())
        binary_event_effects_writer.writerow(binary_event_effects_fields)

    # Process each PDF in the input directory
    # Log start of processing
    logging.info(f"Starting to process PDFs from directory: {input_dir}")
    logging.info(f"Output files will be written to: {timestamped_output_dir}")
    logging.info(f"Processing papers for dependent variable '{dependent_variable}' and independent variables {independent_variables}")

    # Helper function for progress updates
    def update_progress(stage, paper_name=None, progress_pct=0, details=None):
        if progress_callback:
            progress_callback(stage, paper_name, progress_pct, details)

    update_progress("📊 Data Processing Setup", progress_pct=0.25, details="Setting up CSV files and output structure")

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    total_files = len(pdf_files)

    logging.info(f"Processing the following files: {pdf_files}")
    update_progress("🚀 Starting Paper Analysis", progress_pct=0.3, details=f"Beginning analysis of {total_files} papers")

    for file_index, filename in enumerate(pdf_files):
        paper_path = os.path.join(input_dir, filename)
        file_id = os.path.splitext(filename)[0]  # Use as fallback if metadata extraction fails

        # Calculate progress percentage
        file_progress_start = 0.3 + (0.6 * file_index / total_files)
        file_progress_end = 0.3 + (0.6 * (file_index + 1) / total_files)

        logging.info(f"Processing paper: {filename}")
        update_progress("📄 Processing Paper", filename, file_progress_start, "Loading and parsing document")

        try:
            # Invoke the graph
            logging.info(f"Invoking graph analysis for {filename}")
            update_progress("🤖 AI Analysis", filename, file_progress_start + 0.1, "Running AI extraction pipeline")

            output = graph.invoke({
                "paper_path": paper_path,
                "independent_variables": independent_variables,
                "dependent_variable": dependent_variable,
                "user_instructions": user_instructions,
                "effect_types_to_extract": effect_types_to_extract,
                "target_groups_for_comparison": target_groups_for_comparison
            })

            logging.info(f"Output from invoking graph: {output}")
            update_progress("📊 Extracting Data", filename, file_progress_start + 0.2, "Processing extracted information")

            # Extract relevant information
            paper_meta_info: PaperMetaInfo = output.get("paper_meta_info")
            samples_complete_info: List[SampleCompleteInfo] = output.get("samples_complete_info", [])
            html_report: str = output.get("html_report")

            # Extract paper_id from metadata, fallback to filename if not available
            paper_id = getattr(paper_meta_info, "paper_id", file_id) if paper_meta_info else file_id

            # Save HTML report if generated
            if html_report:
                update_progress("📋 Generating Report", filename, file_progress_start + 0.25, "Creating HTML report")
                # Sanitize the paper_id to ensure it's safe for use as a filename
                safe_paper_id = sanitize_filename(paper_id)

                # Ensure we have a valid filename (fallback to file_id if sanitization results in empty string)
                if not safe_paper_id or safe_paper_id.isspace():
                    safe_paper_id = sanitize_filename(file_id) or "unknown_paper"

                html_filename = f"{safe_paper_id}_report.html"
                html_filepath = os.path.join(timestamped_output_dir, html_filename)

                try:
                    with open(html_filepath, 'w', encoding='utf-8') as html_file:
                        html_file.write(html_report)
                    logging.info(f"HTML report saved: {html_filename}")
                except (OSError, IOError) as e:
                    logging.error(f"Failed to save HTML report: {str(e)}")
                    # Try with a simple fallback filename
                    fallback_filename = f"report_{file_id}_{datetime.now().strftime('%H%M%S')}.html"
                    fallback_filepath = os.path.join(timestamped_output_dir, fallback_filename)
                    try:
                        with open(fallback_filepath, 'w', encoding='utf-8') as html_file:
                            html_file.write(html_report)
                        logging.info(f"HTML report saved with fallback filename: {fallback_filename}")
                    except Exception as e2:
                        logging.error(f"Failed to save HTML report even with fallback: {str(e2)}")

            if paper_meta_info and samples_complete_info:
                logging.info(f"Writing data for paper: {paper_meta_info.title}")
                update_progress("💾 Writing Data", filename, file_progress_start + 0.3, f"Saving extracted data for {len(samples_complete_info)} samples")
                # Write paper information including quality control fields
                papers_writer.writerow([
                    paper_id,
                    paper_meta_info.title,
                    paper_meta_info.publication_year,
                    paper_meta_info.journal,
                    paper_meta_info.published_status,
                    paper_meta_info.publication_type,
                    format_list_for_csv(paper_meta_info.authors) if paper_meta_info.authors else "",
                    output.get("paper_relevance", False),
                    paper_meta_info.confidence_level,
                    format_list_for_csv(paper_meta_info.reasons)
                ])

                # Process samples and variables
                logging.info(f"Processing {len(samples_complete_info)} samples from {filename}")
                for sample_id, sample_info in enumerate(samples_complete_info):
                    logging.debug(f"Writing sample {sample_id} data: {sample_info.sample_name}")

                    # Create a row with all sample fields
                    sample_row = [paper_id, sample_id, sample_info.sample_name]

                    # Dynamically add all fields from sample_basic_info
                    for field in SampleBasicInfo.__annotations__.keys():
                        field_value = getattr(sample_info.sample_basic_info, field, None)
                        sample_row.append(format_list_for_csv(field_value))

                    samples_writer.writerow(sample_row)

                    logging.debug(f"Writing {len(sample_info.variables_info)} variables for sample {sample_id}")
                    for var_info in sample_info.variables_info:
                        # Create a row with all variable fields
                        var_row = [paper_id, sample_id]

                        # Dynamically add all fields from variable_info
                        for field in VariableInfoInSample.__annotations__.keys():
                            field_value = getattr(var_info, field, None)
                            var_row.append(format_list_for_csv(field_value))

                        variables_writer.writerow(var_row)

                    # Write correlations only if correlation analysis was selected
                    if correlations_writer:
                        logging.debug(f"Writing {len(sample_info.correlations_info)} correlations for sample {sample_id}")
                        for corr_info in sample_info.correlations_info:
                            # Create a row with all correlation fields
                            corr_row = [paper_id, sample_id]

                            # Dynamically add all fields from correlation_info
                            for field in CorrelationInfoInSample.__annotations__.keys():
                                field_value = getattr(corr_info, field, None)
                                corr_row.append(format_list_for_csv(field_value))

                            correlations_writer.writerow(corr_row)

                    # Write between-group effects data only if selected
                    if between_group_effects_writer:
                        logging.debug(f"Writing {len(sample_info.between_group_effects_info)} between-group effects for sample {sample_id}")
                        for effect_info in sample_info.between_group_effects_info:
                            # Create base row with overall effect information (excluding groups field)
                            base_effect_row = [paper_id, sample_id]

                            # Add all fields from effect_info except 'groups'
                            for field in BetweenGroupEffectInSample.__annotations__.keys():
                                if field != 'groups':
                                    field_value = getattr(effect_info, field, None)
                                    base_effect_row.append(format_list_for_csv(field_value))

                            # If there are groups, create one row per group
                            if effect_info.groups:
                                for group_info in effect_info.groups:
                                    # Combine base effect info with individual group info
                                    full_row = base_effect_row.copy()

                                    # Add all group fields
                                    for group_field in GroupInfo.__annotations__.keys():
                                        group_value = getattr(group_info, group_field, None)
                                        full_row.append(format_list_for_csv(group_value))

                                    between_group_effects_writer.writerow(full_row)
                            else:
                                # No groups found, write row with empty group fields
                                full_row = base_effect_row + [''] * len(GroupInfo.__annotations__.keys())
                                between_group_effects_writer.writerow(full_row)

                    # Write within-subject effects data only if selected
                    if within_subject_effects_writer:
                        logging.debug(f"Writing {len(sample_info.within_subject_effects_info)} within-subject effects for sample {sample_id}")
                        for effect_info in sample_info.within_subject_effects_info:
                            # Create a row with all within-subject effects fields
                            effect_row = [paper_id, sample_id]

                            # Dynamically add all fields from effect_info
                            for field in WithinSubjectEffectInSample.__annotations__.keys():
                                field_value = getattr(effect_info, field, None)
                                effect_row.append(format_list_for_csv(field_value))

                            within_subject_effects_writer.writerow(effect_row)

                    # Write binary event effects data only if selected
                    if binary_event_effects_writer:
                        logging.debug(f"Writing {len(sample_info.binary_event_effects_info)} binary event effects for sample {sample_id}")
                        for effect_info in sample_info.binary_event_effects_info:
                            # Create a row with all binary event effects fields
                            effect_row = [paper_id, sample_id]

                            # Dynamically add all fields from effect_info
                            for field in BinaryEventEffectInSample.__annotations__.keys():
                                field_value = getattr(effect_info, field, None)
                                effect_row.append(format_list_for_csv(field_value))

                            binary_event_effects_writer.writerow(effect_row)

                update_progress("✅ Paper Complete", filename, file_progress_end, f"Successfully processed {paper_meta_info.title}")
            else:
                logging.warning(f"No valid data extracted from {filename}")
                update_progress("⚠️ No Data Extracted", filename, file_progress_end, "Paper processed but no valid data found")
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}", exc_info=True)
            update_progress("❌ Processing Error", filename, file_progress_end, f"Error: {str(e)[:50]}...")

    # Close all files
    logging.info("Closing output files")
    update_progress("📁 Finalizing Results", progress_pct=0.95, details="Closing files and generating final output")

    papers_file.close()
    samples_file.close()
    variables_file.close()

    # Conditionally close effect-specific files
    if correlations_file:
        correlations_file.close()
    if between_group_effects_file:
        between_group_effects_file.close()
    if within_subject_effects_file:
        within_subject_effects_file.close()
    if binary_event_effects_file:
        binary_event_effects_file.close()

    update_progress("🎉 Processing Complete", progress_pct=1.0, details=f"Successfully processed {total_files} papers")
    logging.info("Processing complete")


if __name__ == "__main__":
    # Set up the directories
    input_dir = "data/candidate_papers"
    output_dir = "data/output"
    
    # Define your meta-analysis variables
    dependent_variable = "workplace aggression"
    independent_variables = ["narcissism", "neuroticism"]
    
    # Process the papers
    process_papers(input_dir, output_dir, dependent_variable, independent_variables)
    