import json, os, operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from llama_index import download_loader
from langgraph.graph import END, StateGraph
from langgraph.constants import Send

from .data_types import MetaAnalysisInfo, PaperMetaInfo, SampleInfo, VariableInfo, VariablePairInfo
from utils.helpers import get_llamaparsed_doc
from utils.models import GPT4O_LANGCHAIN_NEW


class GraphState(TypedDict):
    paper_path: str
    paper_content: str
    paper_meta_info: Optional[PaperMetaInfo]
    sample_info: Optional[List[SampleInfo]]
    variable_info: Optional[List[VariableInfo]]
    variable_pair_info: Optional[List[VariablePairInfo]]
    meta_info: Optional[MetaAnalysisInfo]
    independent_variable: str
    dependent_variable: str
    worklog: str

class MetaAnalysisGraph:
    def __init__(self):
        self.llm = OpenAI(temperature=0)

    def load_and_parse_pdf(self, state: GraphState) -> GraphState:
        document = get_llamaparsed_doc(state["paper_path"])
        return {"paper_content": document}

    def extract_paper_meta_info(self, state: GraphState) -> GraphState:
        output_parser = PydanticOutputParser(pydantic_object=PaperMetaInfo)
        
        prompt = PromptTemplate(
            template="Extract meta-analysis information from the following text:\n\n{text}\n\n{format_instructions}",
            input_variables=["text"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(text=state["paper_content"])
        paper_meta_info = output_parser.parse(output)
        
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
        workflow.add_node("extract_paper_meta_info", self.extract_paper_meta_info)
        workflow.add_node("extract_sample_info", self.extract_sample_info)
        workflow.add_node("extract_variable_info", self.extract_variable_info)
        workflow.add_node("extract_variable_pair_info", self.extract_variable_pair_info)
        workflow.add_node("synthesize_meta_info", self.synthesize_meta_info)

        workflow.set_entry_point("load_and_parse_pdf")

        workflow.add_edge("load_and_parse_pdf", "extract_paper_meta_info")
        workflow.add_edge("extract_paper_meta_info", "extract_sample_info")
        workflow.add_edge("extract_sample_info", "extract_variable_info")
        workflow.add_edge("extract_variable_info", "extract_variable_pair_info")
        workflow.add_edge("extract_variable_pair_info", "synthesize_meta_info")
        workflow.add_edge("synthesize_meta_info", END)

        graph = workflow.compile()
        print(graph.get_graph().draw_ascii())
        return graph

if __name__ == "__main__":
    pdf_path = "/path/to/your/pdf/file.pdf"
    independent_variable = "your_independent_variable"
    dependent_variable = "your_dependent_variable"
    
    meta_analysis = MetaAnalysisGraph()
    graph = meta_analysis.create_meta_analysis_graph()
    output = graph.invoke({
        "paper_path": pdf_path,
        "independent_variable": independent_variable,
        "dependent_variable": dependent_variable
    })
    
    print("Meta-Analysis Results:")
    print(json.dumps(output["meta_info"], indent=2))
    print("\nWorklog:")
    print(output["worklog"])
