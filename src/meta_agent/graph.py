import json, os, operator
from typing import Annotated, List, Optional, Literal, Dict
from typing_extensions import TypedDict

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.constants import Send

from .data_types import ( 
    PaperMetaInfo, 
    SampleBasicInfo,
    SampleCompleteInfo
)

from src.utils.helpers import get_llamaparsed_doc, pretty_print_sample_info
from src.utils.models import GPT4O_LANGCHAIN_NEW
from .chains import (
    get_is_paper_relevant_chain, 
    get_extract_paper_meta_info_chain, 
    get_extract_samples_basic_info_chain,
    get_extract_variables_info_from_sample_chain
)

class GraphState(TypedDict):
    paper_path: str  # Path to the paper file
    dependent_variable: str  # The dependent variable being studied
    independent_variables: List[str]  # The independent variables being studied
    paper_content: Annotated[Optional[List[Document]], operator.add] = None  # Content of the paper
    paper_relevance: Annotated[Optional[bool], operator.add] = None  # Boolean indicating if the paper is relevant to the meta-analysis
    paper_meta_info: Annotated[Optional[PaperMetaInfo], operator.add] = None  # Metadata information about the paper
    samples_basic_info: Annotated[Optional[List[SampleBasicInfo]], operator.add] = None  # Basic information about the samples in the paper
    samples_complete_info: Annotated[Optional[List[SampleCompleteInfo]], operator.add] = None  # Complete information about the samples in the paper
    worklog: Annotated[str, operator.add] = ""  # Log of work done by this LangGraph 

class MetaAnalysisGraph:
    def __init__(self):
        self.llm = OpenAI(temperature=0)

    def load_and_parse_pdf(self, state: GraphState) -> GraphState:
        # Add worklog entry for loading and parsing PDF
        worklog_entry = f"Loaded and parsed PDF from {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        print("Worklog:", state["worklog"])
        documents = get_llamaparsed_doc(state["paper_path"])
        print(documents[0].page_content)  # Print the first 100 characters of the document
        return {
            "paper_content": documents, 
            "worklog": state["worklog"]
        }

    def judge_paper_relevance(self, state: GraphState) -> GraphState:
        # Add worklog entry for judging paper relevance
        worklog_entry = f"Judging paper relevance for {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        print("Worklog:", state["worklog"])

        dependent_variable = state["dependent_variable"]
        independent_variables = state["independent_variables"]
        paper_content = state["paper_content"]

        chain = get_is_paper_relevant_chain()
        paper_relevance = chain.invoke(
            {
                "dependent_variable": dependent_variable, 
                "independent_variables": independent_variables, 
                "paper_content": paper_content
            }
        )
        print("Paper Relevance:", paper_relevance.is_relevant)
        return {"paper_relevance": paper_relevance.is_relevant, "worklog": state["worklog"]} 

    def route_based_on_relevance(self, state: GraphState) -> Literal["end", "relevant"]:
        # Add worklog entry for routing based on relevance
        worklog_entry = f"Routing based on relevance for {state['paper_path']}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        print("Worklog:", state["worklog"])

        if state.get("paper_relevance"):
            return "relevant"
        else:
            return "end"

    def extract_paper_meta_info(self, state: GraphState) -> GraphState:
        chain = get_extract_paper_meta_info_chain()
        output = chain.invoke(
            {
                "paper_content": state["paper_content"][:10]
            }
        )
        paper_meta_info = PaperMetaInfo.parse_obj(output)
        print("Paper Meta Info:", paper_meta_info)
        return {"paper_meta_info": paper_meta_info}

    def extract_samples_info(self, state: GraphState) -> GraphState:
        """
            Extract basic sample information from the paper content.
            For each sample, extract the sample name along with the basic information.
        """
        
        worklog_entry = f"Extracting samples info from paper content"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        print("Worklog:", state["worklog"])
        # Extract sample information from the paper content
        chain = get_extract_samples_basic_info_chain()
        output = chain.invoke(
            {
                "paper_content": state["paper_content"]
            }
        )
        samples_basic_info = output.sample_info
        print("Samples Info:", samples_basic_info)
        
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
                    "independent_variables": state["independent_variables"]
                }
            ) 
            for s in state["samples_basic_info"]
        ]


    def extract_variable_info_in_sample(self, state: GraphState) -> GraphState:
        """
            Extract variable information in a sample.
        """

        worklog_entry = f"Extracting variable information in sample {state['sample_basic_info'].sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        print("worklog:", state["worklog"])

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        sample_description = sample_basic_info.sample_description
        sample_size = sample_basic_info.sample_size

        sample_short_description = f"""
            Sample Name: {sample_name}
            Sample Description: {sample_description}
            Sample Size: {sample_size}
        """

        variables = [state["dependent_variable"]] + state["independent_variables"]
        chain = get_extract_variables_info_from_sample_chain()
        output = chain.invoke(
            {
                "paper_content": state["paper_content"],
                "sample_description": sample_short_description,
                "variables": variables
            }
        )

        print("Variables Info:", output)

        sample_complete_info = SampleCompleteInfo(
            sample_name=sample_name,
            sample_basic_info=sample_basic_info,
            variables_info=output.variables_info,
            correlations_info=output.correlations_info
        )

        print("Sample_complete_info: \n\n", sample_complete_info)

        return {
            "samples_complete_info": [sample_complete_info], 
            "worklog": state["worklog"]
        }

    def synthesize_meta_info(self, state: GraphState) -> GraphState:
        current_worklog = "--- Synthesizing the Response ---"

        return {
            "worklog": current_worklog
        }

    def create_meta_analysis_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("load_and_parse_pdf", self.load_and_parse_pdf)
        workflow.add_node("judge_paper_relevance", self.judge_paper_relevance)
        workflow.add_node("extract_paper_meta_info", self.extract_paper_meta_info)
        workflow.add_node("extract_samples_info", self.extract_samples_info)
        workflow.add_node("extract_variable_info_in_sample", self.extract_variable_info_in_sample)
        workflow.add_node("synthesize_meta_info", self.synthesize_meta_info)
        workflow.set_entry_point("load_and_parse_pdf")
        # workflow.add_edge("load_and_parse_pdf", "extract_paper_meta_info")
        # workflow.add_edge("extract_paper_meta_info", "extract_sample_info")
        # workflow.add_edge("extract_sample_info", "extract_variable_info")
        # workflow.add_edge("extract_variable_info", "extract_variable_pair_info")
        # workflow.add_edge("extract_variable_pair_info", "synthesize_meta_info")
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
        print(graph.get_graph().draw_ascii())
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

    print("--- PRINTING AGENT OUTPUT ---\n\n\n\n")

    print("--- Paper Meta Info ---")
    print(output["paper_meta_info"])

    print("--- Samples Complete Info ---")
    for sample in output["samples_complete_info"]:
        pretty_print_sample_info(sample)

    
