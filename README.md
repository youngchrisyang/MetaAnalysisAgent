# Meta-Analysis Agent

This project is an automated meta-analysis agent that processes academic papers to extract and analyze information about workplace behavior and personality traits.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Output](#output)
5. [Appendix: Meta-Agent Description](#appendix-meta-agent-description)

## Getting Started

### Prerequisites

- Git
- Python 3.10 or higher
- Poetry (for dependency management)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/meta-analysis-agent.git
   cd meta-analysis-agent
   ```

2. Set up the Python virtual environment:

   **Option 1: Using Poetry**
   ```
   poetry install
   ```

   Activate the virtual environment:
   ```
   poetry shell
   ```

   **Option 2: Using Conda**
   ```
   conda create -n <your_env_name> python=3.10
   conda activate <your_env_name>
   pip install -r requirements.txt
   ```

   If not the first time, activate the virtual environment already set up:
   ```
   conda activate <your_env_name>
   ```

3. Create a `.env` file in the root directory and add your API keys. Use the `.env_example` file as a template.

## Usage


### Running the Application

You can run the Meta-Analysis Agent in two ways:

#### Option 1: Using the Streamlit Web Interface

For a user-friendly interface, run the Streamlit app:

```
streamlit run app.py
```
This will launch a web interface where you can upload papers, configure analysis parameters, and view results interactively.

#### Option 2: Running the Script Directly

The main script to run is `src/process.py`. Before running, you may want to modify the following variables in the `if __name__ == "__main__":` block:

- `input_dir`: Directory containing the input PDFs (default: "data/candidate_papers")
- `output_dir`: Directory for output CSV files (default: "data/output")
- `dependent_variable`: The dependent variable for your meta-analysis
- `independent_variables`: List of independent variables for your meta-analysis

To run the script, navigate to the root directory of the project (meta-analysis-agent) and execute the following command:

    ```
    python -m src.process
    ```

##### Output

The script generates four CSV files in the `data/output` directory:

1. `papers.csv`: General information about each processed paper
2. `samples.csv`: Information about each sample in the papers
3. `variables.csv`: Details about variables studied in each sample
4. `correlations.csv`: Correlation information between variables


## Appendix: Meta-Agent Description

The meta-analysis agent uses a graph-based approach to process papers:

1. **Load and Parse PDF**: Extracts text content from the PDF.
2. **Judge Paper Relevance**: Determines if the paper is relevant to the meta-analysis.
3. **Extract Paper Meta Info**: Extracts general information about the paper.
4. **Extract Samples Info**: Identifies and extracts information about samples in the study.
5. **Extract Variable Info in Sample**: For each sample, extracts information about variables and their relationships.
6. **Synthesize Meta Info**: Compiles all extracted information into a structured format.

The graph allows for conditional processing and parallel execution of certain steps, improving efficiency and flexibility in handling various paper structures and content.
