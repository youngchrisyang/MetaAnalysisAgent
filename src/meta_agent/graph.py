import json, os, operator
from typing import Annotated, List, Optional, Literal
from typing_extensions import TypedDict

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.constants import Send

from .data_types import MetaAnalysisInfo, PaperMetaInfo, SampleInfo, VariableInfo, VariablePairInfo
from src.utils.helpers import get_llamaparsed_doc
from src.utils.models import GPT4O_LANGCHAIN_NEW
from .chains import (
    get_is_paper_relevant_chain, 
    get_extract_paper_meta_info_chain, 
)

class GraphState(TypedDict):
    paper_path: str  # Path to the paper file
    independent_variable: str  # The independent variables being studied
    dependent_variable: str  # The dependent variable being studied
    paper_content: Optional[List[Document]]  # Content of the paper
    paper_relevance: Optional[bool]  # Boolean indicating if the paper is relevant to the meta-analysis
    paper_meta_info: Optional[PaperMetaInfo]  # Metadata information about the paper
    sample_info: Optional[List[SampleInfo]]  # Information about the sample used in the study
    variable_info: Optional[List[VariableInfo]]  # Information about the variables studied
    variable_pair_info: Optional[List[VariablePairInfo]]  # Information about pairs of variables and their relationships
    meta_info: Optional[MetaAnalysisInfo]  # Overall meta-analysis information
    worklog: str  # Log of work done by this LangGraph 

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
        independent_variable = state["independent_variable"]
        paper_content = state["paper_content"][:5]

        chain = get_is_paper_relevant_chain()
        paper_relevance = chain.invoke(
            {
                "dependent_variable": dependent_variable, 
                "independent_variable": independent_variable, 
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

    def extract_sample_info(self, state: GraphState) -> GraphState:
        output_parser = PydanticOutputParser(pydantic_object=SampleInfo)
        
        prompt = PromptTemplate(
            template="Extract sample information from the following text:\n\n{text}\n\n{format_instructions}",
            input_variables=["text"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(text=state["paper_content"])
        sample_info = output_parser.parse(output)
        
        return {"sample_info": sample_info}

    def extract_variable_info(self, state: GraphState) -> GraphState:
        output_parser = PydanticOutputParser(pydantic_object=VariableInfo)
        
        prompt = PromptTemplate(
            template="Extract variable information from the following text:\n\n{text}\n\n{format_instructions}",
            input_variables=["text"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(text=state["paper_content"])
        variable_info = output_parser.parse(output)
        
        return {"variable_info": variable_info}

    def extract_variable_pair_info(self, state: GraphState) -> GraphState:
        output_parser = PydanticOutputParser(pydantic_object=VariablePairInfo)
        
        prompt = PromptTemplate(
            template="Extract variable pair information from the following text:\n\n{text}\n\n{format_instructions}",
            input_variables=["text"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )   

        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(text=state["paper_content"])
        variable_pair_info = output_parser.parse(output)
        
        return {"variable_pair_info": variable_pair_info}   

    def synthesize_meta_info(self, state: GraphState) -> GraphState:
        current_worklog = "--- Synthesizing the Response ---"

        # Ensemble MetaAnalysisInfo from extracted information
        meta_analysis_info = MetaAnalysisInfo(
            paper_meta_info=state["paper_meta_info"],
            sample_info=state["sample_info"],
            variable_pair_info=state["variable_pair_info"]
        )

        # Update the current worklog
        current_worklog += "\nMeta-analysis information synthesized successfully."

        # Return the assembled MetaAnalysisInfo and updated worklog
        return {
            "meta_analysis_info": meta_analysis_info,
            "worklog": current_worklog
        }

    def create_meta_analysis_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("load_and_parse_pdf", self.load_and_parse_pdf)
        workflow.add_node("judge_paper_relevance", self.judge_paper_relevance)
        workflow.add_node("extract_paper_meta_info", self.extract_paper_meta_info)

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
        workflow.add_edge("extract_paper_meta_info", END)

        graph = workflow.compile()
        print(graph.get_graph().draw_ascii())
        return graph

if __name__ == "__main__":
    pdf_path = "data/test/test_meta_paper.pdf"
    independent_variable = "stereotype lift"
    dependent_variable = "cognitive test performance"
    
    meta_analysis = MetaAnalysisGraph()
    graph = meta_analysis.create_meta_analysis_graph()
    output = graph.invoke({
        "paper_path": pdf_path,
        "independent_variable": independent_variable,
        "dependent_variable": dependent_variable
    })
    