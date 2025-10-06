import json, os, operator, logging
from typing import Annotated, List, Optional, Literal, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.constants import Send
from datetime import datetime

from .data_types import ( 
    PaperMetaInfo, 
    SampleBasicInfo,
    SampleCompleteInfo,
    UserInstructions,
    VariableInfoInSample,
    CorrelationInfoInSample
)

from src.utils.helpers import get_llamaparsed_doc, pretty_print_sample_info
from .chains import (
    get_is_paper_relevant_chain, 
    get_extract_paper_meta_info_chain, 
    get_extract_samples_basic_info_chain,
    get_extract_variables_info_from_sample_chain,
    get_generate_html_report_chain
)
from .sample_graph import SampleGraph
from src.utils.models import (
    LLM_GPT5_FALLBACK,
    LLM_GPT41_FALLBACK,
    LLM_CLAUDE4_FALLBACK,
    LLM_CLAUDE45_FALLBACK
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GraphState(TypedDict):
    paper_path: str  # Path to the paper file
    dependent_variable: str  # The dependent variable being studied
    independent_variables: List[str]  # The independent variables being studied
    effect_types_to_extract: List[str]  # Which effect types to extract
    target_groups_for_comparison: Optional[str]  # Specific groups to compare for between-group analysis
    user_instructions: Optional[UserInstructions]  # User instructions for the meta-analysis
    paper_content: Annotated[Optional[List[Document]], operator.add]  # Content of the paper
    paper_relevance: Annotated[Optional[bool], operator.add]  # Boolean indicating if the paper is relevant to the meta-analysis
    paper_meta_info: Annotated[Optional[PaperMetaInfo], operator.add]  # Metadata information about the paper
    samples_basic_info: Annotated[Optional[List[SampleBasicInfo]], operator.add]  # Basic information about the samples in the paper
    samples_complete_info: Annotated[Optional[List[SampleCompleteInfo]], operator.add]  # Complete information about the samples in the paper
    html_report: Annotated[Optional[str], operator.add]  # Generated HTML report synthesizing all findings
    worklog: Annotated[str, operator.add]  # Log of work done by this LangGraph

class MetaAnalysisGraph:
    def __init__(self, selected_model: str = "LLM_GPT41_FALLBACK", parsing_mode: str = "standard"):
        self.selected_model = selected_model
        self.parsing_mode = parsing_mode

    def get_model_instance(self):
        """Get the actual model instance based on selected_model string"""
        model_mapping = {
            "LLM_GPT5_FALLBACK": LLM_GPT5_FALLBACK,
            "LLM_GPT41_FALLBACK": LLM_GPT41_FALLBACK,
            "LLM_CLAUDE4_FALLBACK": LLM_CLAUDE4_FALLBACK,
            "LLM_CLAUDE45_FALLBACK": LLM_CLAUDE45_FALLBACK
        }
        return model_mapping.get(self.selected_model, LLM_GPT41_FALLBACK)

    def format_data_for_report(self, state: GraphState) -> Dict[str, str]:
        """Format extracted data for LLM report generation"""
        
        paper_meta_info = state.get("paper_meta_info")
        samples_complete_info = state.get("samples_complete_info", [])
        paper_relevance = state.get("paper_relevance", False)
        dependent_variable = state.get("dependent_variable", "")
        independent_variables = state.get("independent_variables", [])
        
        # Format paper information
        paper_info_text = "No paper information extracted."
        if paper_meta_info:
            paper_info_text = f"""
Title: {paper_meta_info.title or 'Not specified'}
Authors: {'; '.join(paper_meta_info.authors) if paper_meta_info.authors else 'Not specified'}
Journal: {paper_meta_info.journal or 'Not specified'}
Publication Year: {paper_meta_info.publication_year or 'Not specified'}
Publication Type: {paper_meta_info.publication_type or 'Not specified'}
Number of Studies: {paper_meta_info.num_studies or 'Not specified'}
Number of Samples: {paper_meta_info.num_samples or 'Not specified'}
Confidence Level: {paper_meta_info.confidence_level}
Reasons: {'; '.join(paper_meta_info.reasons) if paper_meta_info.reasons else 'None'}
"""
        
        # Format samples information
        samples_info_text = "No samples identified."
        if samples_complete_info:
            samples_info_parts = []
            for i, sample in enumerate(samples_complete_info):
                sample_info = sample.sample_basic_info
                sample_text = f"""
Sample {i+1}: {sample.sample_name}
- Description: {sample_info.sample_description}
- Size: {sample_info.sample_size}
- Country: {sample_info.country}
- Sample Type: {sample_info.sample_type}
- Sampling Technique: {sample_info.sampling_technique}
- Mean Age: {sample_info.mean_age or 'Not reported'}
- Age SD: {sample_info.sd_age or 'Not reported'}
- Male Participants: {sample_info.male_n or 'Not reported'}
- Female Participants: {sample_info.female_n or 'Not reported'}
- Response Rate: {sample_info.response_rate or 'Not reported'}%
- Confidence Level: {sample_info.confidence_level}
- Reasons: {'; '.join(sample_info.reasons) if sample_info.reasons else 'None'}
"""
                samples_info_parts.append(sample_text)
            samples_info_text = "\n".join(samples_info_parts)
        
        # Format variables information
        variables_info_text = "No variables extracted."
        if samples_complete_info:
            variables_info_parts = []
            for i, sample in enumerate(samples_complete_info):
                if sample.variables_info:
                    var_list = []
                    for var in sample.variables_info:
                        var_text = f"""
  * {var.variable_name}:
    - Type: {var.variable_type}
    - Scale/Measure: {var.scale_measure or 'Not specified'}
    - Reliability: {var.reliability or 'Not reported'}
    - Mean: {var.mean or 'Not reported'}
    - SD: {var.standard_deviation or 'Not reported'}
    - Confidence: {var.confidence_level}
    - Reasons: {'; '.join(var.reasons) if var.reasons else 'None'}
"""
                        var_list.append(var_text)
                    
                    variables_sample_text = f"""
Variables in Sample {i+1} ({sample.sample_name}):
{''.join(var_list)}"""
                    variables_info_parts.append(variables_sample_text)
            variables_info_text = "\n".join(variables_info_parts) if variables_info_parts else "No variables extracted."
        
        # Format correlations information
        correlations_info_text = "No correlations found."
        if samples_complete_info:
            correlations_info_parts = []
            for i, sample in enumerate(samples_complete_info):
                existing_correlations = sample.correlations_info or []
                if existing_correlations:
                    corr_list = []
                    for corr in existing_correlations:
                        corr_text = f"""
  * {corr.variable1} <-> {corr.variable2}:
    - Correlation: {corr.correlation_coefficient if corr.correlation_coefficient is not None else 'Not reported'}
    - Type: {corr.correlation_type}
    - Significance: {corr.significance_level if corr.significance_level else 'Not reported'}
    - Confidence: {corr.confidence_level}
    - Reasons: {'; '.join(corr.reasons) if corr.reasons else 'None'}
"""
                        corr_list.append(corr_text)
                    
                    correlations_sample_text = f"""
Correlations in Sample {i+1} ({sample.sample_name}):
{''.join(corr_list)}"""
                    correlations_info_parts.append(correlations_sample_text)
                else:
                    correlations_info_parts.append(f"\nNo correlations found in Sample {i+1} ({sample.sample_name})")
            
            correlations_info_text = "\n".join(correlations_info_parts) if correlations_info_parts else "No correlations found."
        
        return {
            "dependent_variable": dependent_variable,
            "independent_variables": ", ".join(independent_variables),
            "paper_relevance": "Yes" if paper_relevance else "No",
            "paper_meta_info": paper_info_text,
            "samples_info": samples_info_text,
            "variables_info": variables_info_text,
            "correlations_info": correlations_info_text
        }

    def load_and_parse_pdf(self, state: GraphState) -> GraphState:
        # Add worklog entry for loading and parsing PDF
        worklog_entry = f"Loaded and parsed PDF from {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])
        
        try:
            documents = get_llamaparsed_doc(state["paper_path"], parsing_mode=self.parsing_mode)
            
            # Check if documents were successfully parsed
            if not documents:
                error_msg = f"Failed to parse PDF: {state['paper_path']} - No content extracted"
                logger.error(error_msg)
                worklog_entry = f"ERROR: {error_msg}"
                state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
                return {
                    "paper_content": [], 
                    "worklog": state["worklog"],
                    "user_instructions": state.get("user_instructions") or UserInstructions()
                }
            
            # Log first part of document content for debugging
            logger.info("Successfully parsed PDF. First 200 characters: %s", 
                       documents[0].page_content[:200] if documents[0].page_content else "No content")
                       
        except Exception as e:
            error_msg = f"Error parsing PDF {state['paper_path']}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            worklog_entry = f"ERROR: {error_msg}"
            state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
            return {
                "paper_content": [], 
                "worklog": state["worklog"],
                "user_instructions": state.get("user_instructions") or UserInstructions()
            }
        
        # Initialize user_instructions if not provided
        if not state.get("user_instructions"):
            state["user_instructions"] = UserInstructions()
            
        return {
            "paper_content": documents, 
            "worklog": state["worklog"],
            "user_instructions": state.get("user_instructions")
        }

    def judge_paper_relevance(self, state: GraphState) -> GraphState:
        # Add worklog entry for judging paper relevance
        worklog_entry = f"Judging paper relevance for {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])

        dependent_variable = state["dependent_variable"]
        independent_variables = state["independent_variables"]
        paper_content = state["paper_content"]
        
        # Check if paper content is empty (parsing failed)
        if not paper_content:
            error_msg = f"Cannot judge relevance - no content available for {state['paper_path']}"
            logger.warning(error_msg)
            worklog_entry = f"WARNING: {error_msg}"
            state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
            # Return False for relevance if no content to analyze
            return {"paper_relevance": False, "worklog": state["worklog"]}
        
        # Get relevance instructions - use user instructions if provided, otherwise use default
        user_instructions = state.get("user_instructions", UserInstructions())
        instructions = user_instructions.paper_relevance_instructions or UserInstructions().paper_relevance_instructions

        try:
            chain = get_is_paper_relevant_chain(llm=self.get_model_instance())
            paper_relevance = chain.invoke(
                {
                    "dependent_variable": dependent_variable,
                    "independent_variables": independent_variables,
                    "paper_content": paper_content,
                    "instructions": instructions
                }
            )
            logger.info("Paper Relevance: %s", paper_relevance.is_relevant)
            return {"paper_relevance": paper_relevance.is_relevant, "worklog": state["worklog"]} 
            
        except Exception as e:
            error_msg = f"Error judging paper relevance for {state['paper_path']}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            worklog_entry = f"ERROR: {error_msg}"
            state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
            # Return False for relevance if processing fails
            return {"paper_relevance": False, "worklog": state["worklog"]}

    def route_based_on_relevance(self, state: GraphState) -> Literal["end", "relevant"]:
        # Add worklog entry for routing based on relevance
        worklog_entry = f"Routing based on relevance for {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])

        if state.get("paper_relevance"):
            return "relevant"
        else:
            return "end"

    def extract_paper_meta_info(self, state: GraphState) -> GraphState:
        worklog_entry = f"Extracting paper meta info for {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])
        
        # Get meta info instructions - use user instructions if provided, otherwise use default
        user_instructions = state.get("user_instructions", UserInstructions())
        instructions = user_instructions.paper_meta_info_instructions or UserInstructions().paper_meta_info_instructions
        
        chain = get_extract_paper_meta_info_chain(llm=self.get_model_instance())
        output = chain({
            "paper_content": state["paper_content"][:10],
            "instructions": instructions
        })
        paper_meta_info = output  # No need to parse_obj since the chain already returns a PaperMetaInfo object
        logger.info("Paper Meta Info: %s", paper_meta_info)
        return {"paper_meta_info": paper_meta_info, "worklog": state["worklog"]}

    def extract_samples_info(self, state: GraphState) -> GraphState:
        """
            Extract basic sample information from the paper content.
            For each sample, extract the sample name along with the basic information.
        """
        
        worklog_entry = f"Extracting samples info from paper content"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])
        
        # Get samples extraction instructions - use user instructions if provided, otherwise use default
        user_instructions = state.get("user_instructions", UserInstructions())
        instructions = user_instructions.samples_extraction_instructions or UserInstructions().samples_extraction_instructions
        
        # Extract sample information from the paper content
        chain = get_extract_samples_basic_info_chain(llm=self.get_model_instance())
        output = chain({
            "paper_content": state["paper_content"],
            "instructions": instructions
        })
        samples_basic_info = output.sample_info
        logger.info("Samples Info: %s", samples_basic_info)
        
        return {
            "samples_basic_info": samples_basic_info,
            "worklog": state["worklog"]
        }

    # Here we define the logic to map out over the generated subjects
    # We will use this an edge in the graph
    def continue_to_extraction_by_sample(self, state: GraphState):
        # We will return a list of `Send` objects
        # Each `Send` object consists of the name of a node in the graph
        # as well as the state to send to that node
        return [
            Send(
                "extract_variable_info_in_sample", 
                {
                    "sample_basic_info": s, 
                    "paper_content": state["paper_content"],
                    "dependent_variable": state["dependent_variable"],
                    "independent_variables": state["independent_variables"],
                    "effect_types_to_extract": state["effect_types_to_extract"],  # FIXED: Direct access, no fallback to all types
                    "target_groups_for_comparison": state.get("target_groups_for_comparison"),
                    "user_instructions": state.get("user_instructions")
                }
            ) 
            for s in state["samples_basic_info"]
        ]

    def extract_variable_info_in_sample(self, state: GraphState) -> GraphState:
        """
        Extract variable information in a sample by invoking the sample graph.
        This method now delegates the actual extraction work to the dedicated sample graph.
        """

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        
        worklog_entry = f"Invoking sample graph for sample {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])

        # Create and invoke the sample graph
        sample_graph = SampleGraph(selected_model=self.selected_model)
        graph = sample_graph.create_sample_graph()
        
        # Prepare input for the sample graph - FIXED: properly pass effect_types_to_extract and target_groups_for_comparison
        sample_graph_input = {
            "sample_basic_info": sample_basic_info,
            "paper_content": state["paper_content"],
            "dependent_variable": state["dependent_variable"],
            "independent_variables": state["independent_variables"],
            "effect_types_to_extract": state["effect_types_to_extract"],  # FIXED: Direct access, no fallback to all types
            "target_groups_for_comparison": state.get("target_groups_for_comparison"),
            "user_instructions": state.get("user_instructions")
        }
        
        logger.info(f"Invoking sample graph for sample: {sample_name}")
        logger.info(f"Effect types to extract: {sample_graph_input['effect_types_to_extract']}")
        if sample_graph_input['target_groups_for_comparison']:
            logger.info(f"Target groups for comparison: {sample_graph_input['target_groups_for_comparison']}")
        
        # Invoke the sample graph
        sample_graph_output = graph.invoke(sample_graph_input)
        
        # Extract the result
        sample_complete_info = sample_graph_output.get("sample_complete_info")
        sample_worklog = sample_graph_output.get("worklog", "")
        
        # Merge worklogs
        combined_worklog = state["worklog"] + f"Sample graph completed for {sample_name}\n" + sample_worklog

        logger.info(f"Sample graph completed for {sample_name}")
        logger.info(f"Sample_complete_info summary: \n- Variables: {[var.variable_name for var in sample_complete_info.variables_info]}\n- Correlations found: {[(c.variable1, c.variable2, c.correlation_coefficient) for c in sample_complete_info.correlations_info]}")

        return {
            "samples_complete_info": [sample_complete_info], 
            "worklog": combined_worklog
        }

    def synthesize_meta_info(self, state: GraphState) -> GraphState:
        worklog_entry = "Generating LLM-based comprehensive HTML report from extracted data"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Worklog: %s", state["worklog"])
        
        # Format data for LLM prompt
        logger.info("Formatting data for LLM report generation...")
        formatted_data = self.format_data_for_report(state)
        
        # Generate HTML report using LLM
        logger.info("Calling LLM to generate HTML report...")
        try:
            # Note: get_generate_html_report_chain always uses Claude 4.5 internally
            html_report_chain = get_generate_html_report_chain()
            html_report = html_report_chain.invoke(formatted_data)

            logger.info(f"LLM-generated HTML report completed (length: {len(html_report)} characters)")

            # Validate HTML completeness - check if it contains essential elements
            if not html_report or len(html_report) < 1000:
                logger.warning(f"HTML report seems incomplete (length: {len(html_report)}), generating fallback")
                html_report = self.generate_fallback_html_report(formatted_data)
            elif not ("</html>" in html_report or "</body>" in html_report):
                logger.warning("HTML report missing closing tags, attempting to fix")
                # Try to close the HTML properly
                if not html_report.endswith("</html>"):
                    if not html_report.endswith("</body>"):
                        html_report += "\n</div>\n</body>\n</html>"
                    else:
                        html_report += "\n</html>"

        except Exception as e:
            logger.error(f"Error generating HTML report with LLM: {str(e)}")
            logger.info("Generating fallback HTML report...")
            html_report = self.generate_fallback_html_report(formatted_data)

        return {
            "html_report": html_report,
            "worklog": state["worklog"]
        }

    def generate_fallback_html_report(self, formatted_data: Dict[str, str]) -> str:
        """Generate a fallback HTML report when LLM generation fails"""
        logger.info("Generating fallback HTML report using template")

        # Extract key information
        dependent_variable = formatted_data.get("dependent_variable", "Unknown")
        independent_variables = formatted_data.get("independent_variables", "Unknown")
        paper_relevance = formatted_data.get("paper_relevance", "Unknown")
        paper_meta_info = formatted_data.get("paper_meta_info", "No information available")
        samples_info = formatted_data.get("samples_info", "No samples information available")
        variables_info = formatted_data.get("variables_info", "No variables information available")
        correlations_info = formatted_data.get("correlations_info", "No correlations information available")

        # Generate comprehensive fallback HTML
        html_report = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meta-Analysis Extraction Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        header {{
            border-bottom: 4px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #3498db;
        }}
        h3 {{
            color: #34495e;
            font-size: 18px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .meta-info {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 25px;
        }}
        .meta-info p {{
            margin: 5px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
        }}
        .badge-relevant {{
            background-color: #27ae60;
            color: white;
        }}
        .badge-not-relevant {{
            background-color: #e74c3c;
            color: white;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
            color: #856404;
        }}
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Meta-Analysis Extraction Report</h1>
            <div class="meta-info">
                <p><strong>Dependent Variable:</strong> {dependent_variable}</p>
                <p><strong>Independent Variables:</strong> {independent_variables}</p>
                <p><strong>Paper Relevance:</strong> {"✅ Relevant" if str(paper_relevance).lower() == "true" else "❌ Not Relevant"}
                   <span class="badge {'badge-relevant' if str(paper_relevance).lower() == "true" else 'badge-not-relevant'}">
                       {"RELEVANT" if str(paper_relevance).lower() == "true" else "NOT RELEVANT"}
                   </span>
                </p>
            </div>
        </header>

        <div class="warning">
            <strong>⚠️ Note:</strong> This report was generated using a fallback template due to an issue with the advanced AI report generator.
            The data below represents the raw extracted information and may require manual interpretation.
        </div>

        <div class="section">
            <h2>📄 Paper Information</h2>
            <pre>{paper_meta_info}</pre>
        </div>

        <div class="section">
            <h2>👥 Samples Information</h2>
            <pre>{samples_info}</pre>
        </div>

        <div class="section">
            <h2>📊 Variables Information</h2>
            <pre>{variables_info}</pre>
        </div>

        <div class="section">
            <h2>🔗 Correlations Information</h2>
            <pre>{correlations_info}</pre>
        </div>

        <div class="section">
            <h2>💡 Recommendations</h2>
            <h3>Next Steps for Researchers:</h3>
            <ul>
                <li><strong>Data Verification:</strong> Review the extracted information for accuracy and completeness</li>
                <li><strong>Quality Assessment:</strong> Check confidence levels and reasons for any concerns</li>
                <li><strong>Manual Review:</strong> Verify correlations and sample characteristics against the original paper</li>
                <li><strong>Meta-Analysis Integration:</strong> Use this extracted data as input for your meta-analysis calculations</li>
            </ul>

            <h3>Quality Control Notes:</h3>
            <ul>
                <li>This extraction was performed automatically using AI agents</li>
                <li>All extractions should be validated by qualified researchers</li>
                <li>Pay special attention to confidence levels and extraction reasons</li>
                <li>Consider re-extracting if data appears incomplete or inconsistent</li>
            </ul>
        </div>

        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666;">
            <p>Generated by Meta-Analysis Agent | Academic Research Tool</p>
            <p><small>Report created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </footer>
    </div>
</body>
</html>'''

        return html_report

    def create_meta_analysis_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("load_and_parse_pdf", self.load_and_parse_pdf)
        workflow.add_node("judge_paper_relevance", self.judge_paper_relevance)
        workflow.add_node("extract_paper_meta_info", self.extract_paper_meta_info)
        workflow.add_node("extract_samples_info", self.extract_samples_info)
        workflow.add_node("extract_variable_info_in_sample", self.extract_variable_info_in_sample)
        workflow.add_node("synthesize_meta_info", self.synthesize_meta_info)
        workflow.set_entry_point("load_and_parse_pdf")
        
        workflow.add_edge("load_and_parse_pdf", "judge_paper_relevance")
        workflow.add_conditional_edges(
            "judge_paper_relevance", 
            self.route_based_on_relevance,
            {
                "relevant": "extract_paper_meta_info", 
                "end": END
            }
        )
        
        workflow.add_edge("extract_paper_meta_info", "extract_samples_info")
        workflow.add_conditional_edges("extract_samples_info", self.continue_to_extraction_by_sample, ["extract_variable_info_in_sample"])

        workflow.add_edge("extract_variable_info_in_sample", "synthesize_meta_info")         
        workflow.add_edge("synthesize_meta_info", END)

        graph = workflow.compile()
        logger.info(graph.get_graph().draw_ascii())
        return graph

if __name__ == "__main__":
    # pdf_path = "data/test/test_simple.pdf"
    # independent_variables = ["Narcissism", "Hostility"]
    # dependent_variable = "Verbal Aggression"
    
    pdf_path = "data/test/test_complicated.pdf"
    independent_variables = ["neuroticism", "narcissism"]
    dependent_variable = "individual undermining"
    
    meta_analysis = MetaAnalysisGraph()
    graph = meta_analysis.create_meta_analysis_graph()
    output = graph.invoke({
        "paper_path": pdf_path,
        "independent_variables": independent_variables,
        "dependent_variable": dependent_variable
    })    

    logger.info("--- PRINTING AGENT OUTPUT ---\n\n\n\n")

    logger.info("--- Paper Meta Info ---")
    logger.info(output["paper_meta_info"])

    logger.info("--- Samples Complete Info ---")
    for sample in output["samples_complete_info"]:
        pretty_print_sample_info(sample)
