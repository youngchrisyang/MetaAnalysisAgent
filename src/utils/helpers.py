import os, json
from dotenv import load_dotenv, find_dotenv
from llama_parse import LlamaParse
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

def get_llamaparsed_doc(file_path):
    document = LlamaParse(result_type="markdown", language='en').load_data(file_path)
    document = [doc.to_langchain_format() for doc in document]
    return document
    
