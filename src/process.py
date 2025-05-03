import os
import csv
import logging
from datetime import datetime
from typing import List, Dict
from src.meta_agent.graph import MetaAnalysisGraph
from src.meta_agent.data_types import FinalMetaAnalysisInfo, SampleCompleteInfo, PaperMetaInfo, VariableInfoInSample, CorrelationInfoInSample, SampleBasicInfo
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

def process_papers(input_dir: str, output_dir: str, dependent_variable: str, independent_variables: List[str], user_instructions=None):
    meta_analysis = MetaAnalysisGraph()
    graph = meta_analysis.create_meta_analysis_graph()

    # Prepare output CSV files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamped_output_dir, exist_ok=True)

    papers_file = open(os.path.join(timestamped_output_dir, 'papers.csv'), 'w', newline='', encoding='utf-8')
    samples_file = open(os.path.join(timestamped_output_dir, 'samples.csv'), 'w', newline='', encoding='utf-8')
    variables_file = open(os.path.join(timestamped_output_dir, 'variables.csv'), 'w', newline='', encoding='utf-8')
    correlations_file = open(os.path.join(timestamped_output_dir, 'correlations.csv'), 'w', newline='', encoding='utf-8')

    papers_writer = csv.writer(papers_file)
    samples_writer = csv.writer(samples_file)
    variables_writer = csv.writer(variables_file)
    correlations_writer = csv.writer(correlations_file)

    # Write headers dynamically based on the model fields
    papers_writer.writerow(['paper_id', 'title', 'publication_year', 'journal', 'published_status', 'publication_type', 'is_relevant'])
    
    # Get sample fields dynamically from the model
    sample_fields = ['paper_id', 'sample_id', 'sample_name'] + list(SampleBasicInfo.__annotations__.keys())
    samples_writer.writerow(sample_fields)
    
    # Get variable fields dynamically
    variable_fields = ['paper_id', 'sample_id'] + list(VariableInfoInSample.__annotations__.keys())
    variables_writer.writerow(variable_fields)
    
    # Get correlation fields dynamically
    correlation_fields = ['paper_id', 'sample_id'] + list(CorrelationInfoInSample.__annotations__.keys())
    correlations_writer.writerow(correlation_fields)

    # Process each PDF in the input directory
    # Log start of processing
    logging.info(f"Starting to process PDFs from directory: {input_dir}")
    logging.info(f"Output files will be written to: {timestamped_output_dir}")
    logging.info(f"Processing papers for dependent variable '{dependent_variable}' and independent variables {independent_variables}")

    logging.info(f"Processing the following files: {os.listdir(input_dir)}")
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            paper_path = os.path.join(input_dir, filename)
            file_id = os.path.splitext(filename)[0]  # Use as fallback if metadata extraction fails

            logging.info(f"Processing paper: {filename}")
            try:
                # Invoke the graph
                logging.info(f"Invoking graph analysis for {filename}")
                output = graph.invoke({
                    "paper_path": paper_path,
                    "independent_variables": independent_variables,
                    "dependent_variable": dependent_variable,
                    "user_instructions": user_instructions
                })

                logging.info(f"Output from invoking graph: {output}")

                # Extract relevant information
                paper_meta_info: PaperMetaInfo = output.get("paper_meta_info")
                samples_complete_info: List[SampleCompleteInfo] = output.get("samples_complete_info", [])

                # Extract paper_id from metadata, fallback to filename if not available
                paper_id = getattr(paper_meta_info, "paper_id", file_id) if paper_meta_info else file_id
                
                if paper_meta_info and samples_complete_info:
                    logging.info(f"Writing data for paper: {paper_meta_info.title}")
                    # Write paper information
                    papers_writer.writerow([
                        paper_id,
                        paper_meta_info.title,
                        paper_meta_info.publication_year,
                        paper_meta_info.journal,
                        paper_meta_info.published_status,
                        paper_meta_info.publication_type,
                        output.get("paper_relevance", False)
                    ])

                    # Process samples and variables
                    logging.info(f"Processing {len(samples_complete_info)} samples from {filename}")
                    for sample_id, sample_info in enumerate(samples_complete_info):
                        logging.debug(f"Writing sample {sample_id} data: {sample_info.sample_name}")
                        
                        # Create a row with all sample fields
                        sample_row = [paper_id, sample_id, sample_info.sample_name]
                        
                        # Dynamically add all fields from sample_basic_info
                        for field in SampleBasicInfo.__annotations__.keys():
                            sample_row.append(getattr(sample_info.sample_basic_info, field, None))
                        
                        samples_writer.writerow(sample_row)

                        logging.debug(f"Writing {len(sample_info.variables_info)} variables for sample {sample_id}")
                        for var_info in sample_info.variables_info:
                            # Create a row with all variable fields
                            var_row = [paper_id, sample_id]
                            
                            # Dynamically add all fields from variable_info
                            for field in VariableInfoInSample.__annotations__.keys():
                                var_row.append(getattr(var_info, field, None))
                                
                            variables_writer.writerow(var_row)

                        logging.debug(f"Writing {len(sample_info.correlations_info)} correlations for sample {sample_id}")
                        for corr_info in sample_info.correlations_info:
                            # Create a row with all correlation fields
                            corr_row = [paper_id, sample_id]
                            
                            # Dynamically add all fields from correlation_info
                            for field in CorrelationInfoInSample.__annotations__.keys():
                                corr_row.append(getattr(corr_info, field, None))
                                
                            correlations_writer.writerow(corr_row)
                else:
                    logging.warning(f"No valid data extracted from {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}", exc_info=True)

    # Close all files
    logging.info("Closing output files")
    papers_file.close()
    samples_file.close()
    variables_file.close()
    correlations_file.close()
    logging.info("Processing complete")

if __name__ == "__main__":
    input_dir = "data/candidate_papers"
    output_dir = "data/output"
    dependent_variable = "Honesty-Humility"
    independent_variables = ["leader effectiveness","leader emergence"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    process_papers(input_dir, output_dir, dependent_variable, independent_variables)
    logging.info(f"Processing complete. Output files are in: {output_dir}")