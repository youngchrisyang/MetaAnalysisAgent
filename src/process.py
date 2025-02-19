import os
import csv
import logging
from datetime import datetime
from typing import List, Dict
from src.meta_agent.graph import MetaAnalysisGraph
from src.meta_agent.data_types import FinalMetaAnalysisInfo, SampleCompleteInfo, PaperMetaInfo, VariableInfoInSample, CorrelationInfoInSample
from src.utils.initialization import initialize_env

initialize_env()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def process_papers(input_dir: str, output_dir: str, dependent_variable: str, independent_variables: List[str]):
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

    # Write headers
    papers_writer.writerow(['paper_id', 'title', 'publication_year', 'journal', 'published_status', 'publication_type', 'is_relevant'])
    samples_writer.writerow(['paper_id', 'sample_id', 'sample_name', 'sample_size', 'sample_description', 'country', 'sampling_technique', 'sample_type', 'mean_age', 'sd_age', 'male_n', 'female_n', 'major_ethnicity', 'major_ethnicity_percentage', 'response_rate'])
    variables_writer.writerow(['paper_id', 'sample_id', 'variable_name', 'variable_type', 'scale_measure', 'reliability', 'mean', 'standard_deviation'])
    correlations_writer.writerow(['paper_id', 'sample_id', 'variable1', 'variable2', 'exists', 'correlation_coefficient'])

    # Process each PDF in the input directory

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            paper_path = os.path.join(input_dir, filename)
            paper_id = os.path.splitext(filename)[0]  # Use filename without extension as paper_id

            try:
                # Invoke the graph
                output = graph.invoke({
                    "paper_path": paper_path,
                    "independent_variables": independent_variables,
                    "dependent_variable": dependent_variable
                })

                logging.info(f"output from invoking graph: {output}")

                # Extract relevant information
                paper_meta_info: PaperMetaInfo = output.get("paper_meta_info")
                samples_complete_info: List[SampleCompleteInfo] = output.get("samples_complete_info", [])

                if paper_meta_info and samples_complete_info:
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
                    for sample_id, sample_info in enumerate(samples_complete_info):
                        samples_writer.writerow([
                            paper_id,
                            sample_id,
                            sample_info.sample_name,
                            sample_info.sample_basic_info.sample_size,
                            sample_info.sample_basic_info.sample_description,
                            sample_info.sample_basic_info.country,
                            sample_info.sample_basic_info.sampling_technique,
                            sample_info.sample_basic_info.sample_type,
                            sample_info.sample_basic_info.mean_age,
                            sample_info.sample_basic_info.sd_age,
                            sample_info.sample_basic_info.male_n,
                            sample_info.sample_basic_info.female_n,
                            sample_info.sample_basic_info.major_ethnicity,
                            sample_info.sample_basic_info.major_ethnicity_percentage,
                            sample_info.sample_basic_info.response_rate
                        ])

                        for var_info in sample_info.variables_info:
                            variables_writer.writerow([
                                paper_id,
                                sample_id,
                                var_info.variable_name,
                                var_info.variable_type,
                                var_info.scale_measure,
                                var_info.reliability,
                                var_info.mean,
                                var_info.standard_deviation
                            ])

                        for corr_info in sample_info.correlations_info:
                            correlations_writer.writerow([
                                paper_id,
                                sample_id,
                                corr_info.variable_pair[0],
                                corr_info.variable_pair[1],
                                corr_info.exists,
                                corr_info.correlation_coefficient
                            ])
                else:
                    logging.warning(f"No valid data extracted from {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

    # Close all files
    papers_file.close()
    samples_file.close()
    variables_file.close()
    correlations_file.close()

if __name__ == "__main__":
    input_dir = "data/candidate_papers"
    output_dir = "data/output"
    dependent_variable = "Honesty-Humility"
    independent_variables = ["leader effectiveness","leader emergence"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    process_papers(input_dir, output_dir, dependent_variable, independent_variables)
    logging.info(f"Processing complete. Output files are in: {output_dir}")