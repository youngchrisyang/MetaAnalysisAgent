## Configs used in the meta agent

DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS = """
1. As long as the paper has both of the variables in the data, the paper is relevant.
2. It should be considered relevant if the variables are not directly mentioned but generally related to the research topic.
3. It's possible both variables appeared as the independent variables in the paper provided.
"""

DEFAULT_PAPER_META_INFO_INSTRUCTIONS = """
<empty>
"""

DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS = """
1. First determine how many different samples were used in the paper.
2. For each sample, first provide a name and description to the sample. Then extract the basic information about the sample.
3. Please try your best to extract the information. If you are not sure about the information, you can leave it blank. Please do not make up information.
"""

DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS = """
<empty>
"""