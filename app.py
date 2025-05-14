import os
import tempfile
import streamlit as st
from datetime import datetime
import pandas as pd
import logging
from src.process import process_papers
from src.utils.initialization import initialize_env

# Initialize environment variables
initialize_env()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Meta-Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more academic look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif;
    }
    .stButton button {
        background-color: #2c3e50;
        color: white;
    }
    .output-section {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Meta-Analysis Agent")
st.markdown("""
This application helps researchers conduct meta-analyses by automatically extracting 
information from academic papers. Upload your papers, specify your variables of interest, 
and let the AI do the work.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Analysis Parameters")
    
    # File uploader for PDFs
    uploaded_files = st.file_uploader(
        "Upload academic papers (PDF format)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select multiple PDF files containing academic papers for analysis"
    )
    
    # Variables input
    dependent_variable = st.text_input(
        "Dependent Variable",
        help="The main outcome variable you're studying (e.g., 'Honesty-Humility')"
    )
    
    independent_variables_input = st.text_input(
        "Independent Variables (comma-separated)",
        help="Variables that might influence the dependent variable (e.g., 'leader effectiveness, leader emergence')"
    )
    
    # Custom instructions section
    with st.expander("Advanced: Custom Instructions"):
        st.markdown("""
        You can customize the instructions for each step of the meta-analysis process.
        The default instructions are shown below - modify them as needed.
        """)
        
        # Import default instructions from configs
        from src.utils.configs import (
            DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS,
            DEFAULT_PAPER_META_INFO_INSTRUCTIONS,
            DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS,
            DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS
        )
        
        paper_relevance_instructions = st.text_area(
            "Paper Relevance Instructions",
            value=DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS,
            help="Instructions for determining if a paper is relevant to your meta-analysis"
        )
        
        paper_meta_info_instructions = st.text_area(
            "Paper Meta Information Instructions",
            value=DEFAULT_PAPER_META_INFO_INSTRUCTIONS,
            help="Instructions for extracting paper metadata"
        )
        
        samples_extraction_instructions = st.text_area(
            "Samples Extraction Instructions",
            value=DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS,
            help="Instructions for extracting sample information"
        )
        
        variables_extraction_instructions = st.text_area(
            "Variables Extraction Instructions",
            value=DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS,
            help="Instructions for extracting variable information"
        )
    
    # Process button
    process_button = st.button("Process Papers", type="primary")
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses AI to extract and analyze information from academic papers,
    helping researchers conduct meta-analyses more efficiently.
    """)

# Main content area
if not uploaded_files:
    # Show instructions when no files are uploaded
    st.info("Please upload PDF files using the sidebar to begin analysis.")
    
    # Example of what the tool does
    with st.expander("How it works"):
        st.markdown("""
        1. **Upload Papers**: Select multiple PDF files containing academic papers.
        2. **Specify Variables**: Enter your dependent variable and independent variables.
        3. **Process**: Click the 'Process Papers' button to start the analysis.
        4. **Results**: View and download the extracted data in CSV format.
        
        The system will extract:
        - Paper metadata (title, authors, publication year, etc.)
        - Sample information (size, demographics, etc.)
        - Variables and their measurements
        - Correlation data between variables
        """)
        
        # Sample output preview
        st.subheader("Sample Output Preview")
        sample_df = pd.DataFrame({
            'paper_id': ['paper1', 'paper1', 'paper2'],
            'variable1': ['Honesty-Humility', 'Honesty-Humility', 'Honesty-Humility'],
            'variable2': ['leader effectiveness', 'leader emergence', 'leader effectiveness'],
            'correlation_coefficient': [0.32, 0.18, 0.41]
        })
        st.dataframe(sample_df)

# Process the papers when the button is clicked
if uploaded_files and process_button:
    if not dependent_variable:
        st.error("Please specify a dependent variable.")
    elif not independent_variables_input:
        st.error("Please specify at least one independent variable.")
    else:
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Parse independent variables
        independent_variables = [var.strip() for var in independent_variables_input.split(',')]
        
        # Create user instructions object if any custom instructions were provided
        user_instructions = None
        if (paper_relevance_instructions or paper_meta_info_instructions or 
            samples_extraction_instructions or variables_extraction_instructions):
            from src.meta_agent.data_types import UserInstructions
            user_instructions = UserInstructions(
                paper_relevance_instructions=paper_relevance_instructions or None,
                paper_meta_info_instructions=paper_meta_info_instructions or None,
                samples_extraction_instructions=samples_extraction_instructions or None,
                variables_extraction_instructions=variables_extraction_instructions or None
            )
            st.write("Using custom instructions for analysis")
        
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to the temporary directory
            status_text.text(f"Saving {len(uploaded_files)} uploaded files...")
            
            # Log the files being processed
            file_names = []
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                file_names.append(uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                progress_bar.progress((i + 1) / (len(uploaded_files) * 2))  # First half of progress for saving files
            
            # Log the files in the temp directory
            logger.info(f"Files in temp directory: {os.listdir(temp_dir)}")
            st.write(f"Processing {len(file_names)} files: {', '.join(file_names)}")
            
            # Process the papers
            status_text.text("Processing papers with AI analysis...")
            try:
                # Call the process_papers function with the temporary directory
                # Make sure all files in the directory are being processed
                result = process_papers(temp_dir, "data/output", dependent_variable, independent_variables, user_instructions)
                logger.info(f"Process papers result: {result}")
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Get the timestamp of the most recent output directory
                timestamp_dirs = [d for d in os.listdir("data/output") if os.path.isdir(os.path.join("data/output", d))]
                if timestamp_dirs:
                    latest_dir = max(timestamp_dirs)
                    result_dir = os.path.join("data/output", latest_dir)
                    
                    # Store results in session state
                    st.session_state.result_dir = result_dir
                    st.session_state.papers_df = pd.read_csv(os.path.join(result_dir, "papers.csv"))
                    st.session_state.samples_df = pd.read_csv(os.path.join(result_dir, "samples.csv"))
                    st.session_state.variables_df = pd.read_csv(os.path.join(result_dir, "variables.csv"))
                    st.session_state.correlations_df = pd.read_csv(os.path.join(result_dir, "correlations.csv"))
                    
                    # Display success message
                    st.success(f"Analysis completed successfully! Results saved to: {result_dir}")
                    
                    # Log the number of papers processed
                    num_papers = len(st.session_state.papers_df)
                    logger.info(f"Processed {num_papers} papers")
                    st.write(f"Successfully processed {num_papers} papers")
                else:
                    st.warning("Output directory was created but no results were found.")
            
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.exception(e)
                logger.exception("Error during processing")

# Display results if they exist in session state
if 'result_dir' in st.session_state:
    # Display the results
    st.subheader("Analysis Results")
    
    # Create tabs for different result files
    tab1, tab2, tab3, tab4 = st.tabs(["Papers", "Samples", "Variables", "Correlations"])
    
    # Display each dataframe from session state
    with tab1:
        st.dataframe(st.session_state.papers_df, use_container_width=True)
        st.download_button(
            "Download Papers Data",
            st.session_state.papers_df.to_csv(index=False).encode('utf-8'),
            "papers.csv",
            "text/csv",
            key='download-papers'
        )
    
    with tab2:
        st.dataframe(st.session_state.samples_df, use_container_width=True)
        st.download_button(
            "Download Samples Data",
            st.session_state.samples_df.to_csv(index=False).encode('utf-8'),
            "samples.csv",
            "text/csv",
            key='download-samples'
        )
    
    with tab3:
        st.dataframe(st.session_state.variables_df, use_container_width=True)
        st.download_button(
            "Download Variables Data",
            st.session_state.variables_df.to_csv(index=False).encode('utf-8'),
            "variables.csv",
            "text/csv",
            key='download-variables'
        )
    
    with tab4:
        st.dataframe(st.session_state.correlations_df, use_container_width=True)
        st.download_button(
            "Download Correlations Data",
            st.session_state.correlations_df.to_csv(index=False).encode('utf-8'),
            "correlations.csv",
            "text/csv",
            key='download-correlations'
        )
    
    # Add option to download all CSVs as a zip file
    import io
    import zipfile
    
    # Create a download button for all CSVs in a zip file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        # Add each CSV to the zip file
        zip_file.writestr('papers.csv', st.session_state.papers_df.to_csv(index=False))
        zip_file.writestr('samples.csv', st.session_state.samples_df.to_csv(index=False))
        zip_file.writestr('variables.csv', st.session_state.variables_df.to_csv(index=False))
        zip_file.writestr('correlations.csv', st.session_state.correlations_df.to_csv(index=False))
    
    buffer.seek(0)
    st.download_button(
        label="Download All Data (ZIP)",
        data=buffer,
        file_name="meta_analysis_results.zip",
        mime="application/zip",
        key='download-all'
    )
    
    # Basic visualization if there's correlation data
    if not st.session_state.correlations_df.empty and 'correlation_coefficient' in st.session_state.correlations_df.columns:
        st.subheader("Correlation Visualization")
        # Filter out rows with missing correlation values
        viz_df = st.session_state.correlations_df.dropna(subset=['correlation_coefficient'])
        if not viz_df.empty:
            # Create a heatmap of correlations
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Pivot the data to create a correlation matrix
            pivot_df = viz_df.pivot_table(
                index='variable1', 
                columns='variable2', 
                values='correlation_coefficient',
                aggfunc='mean'  # In case of duplicates, take the mean
            )
            
            # Create the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(pivot_df, dtype=bool))
            heatmap = sns.heatmap(
                pivot_df, 
                annot=True,  # Show correlation values
                cmap="coolwarm",  # Red-blue colormap
                vmin=-1, vmax=1,  # Correlation range
                center=0,  # Center the colormap at zero
                square=True,  # Make cells square
                linewidths=.5,  # Add grid lines
                cbar_kws={"shrink": .8},  # Colorbar settings
                fmt=".2f",  # Format for annotation (2 decimal places)
                mask=mask  # Only show lower triangle
            )
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            
            # Display the heatmap in Streamlit
            st.pyplot(fig)
            
            # Also provide a table view option
            with st.expander("View Correlation Table"):
                st.dataframe(pivot_df, use_container_width=True)
        else:
            st.info("No correlation data available for visualization.")

# Footer
st.markdown("---")
st.markdown(
    "© 2023 Meta-Analysis Agent | Developed for academic research purposes"
) 