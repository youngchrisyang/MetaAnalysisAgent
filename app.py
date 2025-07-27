import os
import tempfile
import streamlit as st
from datetime import datetime
import pandas as pd
import logging
import glob
from src.process import process_papers
from src.utils.initialization import initialize_env
import io
import zipfile

def display_professional_table(df, title="Data", show_summary=True):
    """
    Display a DataFrame in a professional format with proper styling and scrolling.
    
    Args:
        df: pandas DataFrame to display
        title: Title for the table section
        show_summary: Whether to show summary statistics
    """
    if df is None or df.empty:
        st.warning(f"📋 No {title.lower()} data available.")
        return
    
    # Summary statistics
    if show_summary:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", len(df))
        with col2:
            st.metric("📋 Columns", len(df.columns))
        with col3:
            # Calculate memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memory", f"{memory_mb:.2f} MB")
    
    # Professional table display
    st.markdown(f"### 📋 {title}")
    
    # Use st.table() as primary method (we know this works) with intelligent display
    try:
        # Prepare the dataframe for better display
        display_df = df.copy()
        
        # For very wide tables, show key columns first
        if len(df.columns) > 8:
            # Identify key columns to show first
            key_columns = []
            other_columns = []
            
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['id', 'name', 'title', 'description']):
                    key_columns.append(col)
                else:
                    other_columns.append(col)
            
            # Reorder columns: key columns first, then others
            display_df = display_df[key_columns + other_columns]
            
            st.info(f"📊 Showing {len(df.columns)} columns. Key information displayed first.")
        
        # For large datasets, show pagination
        if len(df) > 15:
            st.info(f"📄 Showing first 15 of {len(df)} records. Download CSV for complete data.")
            st.table(display_df.head(15))
        else:
            st.table(display_df)
            
        # Add usage tips
        if len(df.columns) > 5:
            st.markdown("💡 **Tip**: Table may scroll horizontally. Download CSV for full data manipulation.")
        
    except Exception as e:
        st.error(f"Primary table display failed: {str(e)}")
        
        # Fallback 1: Try basic st.table with fewer rows
        try:
            st.markdown("**Using simplified table display:**")
            st.table(df.head(10))
        except Exception as e2:
            st.error(f"Simplified table failed: {str(e2)}")
            
            # Final fallback: Display data as text
            st.markdown("**Data Summary:**")
            st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns**: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            # Show first few records as expandable sections
            for i in range(min(3, len(df))):
                with st.expander(f"Record {i+1}"):
                    for col in df.columns:
                        value = str(df.iloc[i][col])
                        if len(value) > 200:
                            value = value[:200] + "..."
                        st.write(f"**{col}**: {value}")

# Streamlit configuration

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
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional academic look
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        padding: 0.5rem 0;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        font-family: 'Times New Roman', Times, serif;
        color: #1f2937;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 1rem;
        line-height: 1.6;
        max-width: 800px;
    }
    .workflow-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .step-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
    }
    .step-container:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    .step-container::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #3b82f6, #1d4ed8);
        border-radius: 2px 0 0 2px;
    }
    .step-number {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
        margin-right: 1rem;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    .step-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.8rem;
    }
    .step-description {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1rem;
        font-style: italic;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .success-banner {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .reset-btn {
        background: #ef4444 !important;
        color: white !important;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
    }
    .requirements-list {
        background: #f3f4f6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    .variable-focus {
        background: #eff6ff;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1e40af;
        font-weight: 500;
    }
    .header-with-toggle {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }
    .toggle-info-btn {
        background: linear-gradient(135deg, #6366f1, #4338ca) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
        margin: 0 !important;
    }
    
    /* ENHANCED TABLE STYLING FOR BETTER COLUMN WIDTHS */
    div[data-testid="stTable"] {
        width: 100% !important;
        overflow-x: auto !important;
    }
    
    div[data-testid="stTable"] table {
        width: 100% !important;
        table-layout: auto !important;
        border-collapse: collapse !important;
        font-size: 14px !important;
        margin: 0 !important;
    }
    
    div[data-testid="stTable"] th {
        background-color: #f8fafc !important;
        font-weight: 600 !important;
        color: #374151 !important;
        padding: 12px 16px !important;
        text-align: left !important;
        border-bottom: 2px solid #e5e7eb !important;
        white-space: nowrap !important;
        min-width: 120px !important;
        max-width: none !important;
    }
    
    div[data-testid="stTable"] td {
        padding: 10px 16px !important;
        border-bottom: 1px solid #f3f4f6 !important;
        vertical-align: top !important;
        line-height: 1.4 !important;
        word-wrap: break-word !important;
        min-width: 120px !important;
        max-width: none !important;
    }
    
    /* Specific column width adjustments */
    div[data-testid="stTable"] th:nth-child(1),
    div[data-testid="stTable"] td:nth-child(1) {
        min-width: 100px !important; /* ID columns */
        max-width: 150px !important;
    }
    
    div[data-testid="stTable"] th:nth-child(2),
    div[data-testid="stTable"] td:nth-child(2) {
        min-width: 150px !important; /* Name columns */
        max-width: 200px !important;
    }
    
    div[data-testid="stTable"] th:nth-child(3),
    div[data-testid="stTable"] td:nth-child(3) {
        min-width: 150px !important; /* Name columns */
        max-width: 200px !important;
    }
    
    /* Description and content columns - wider */
    div[data-testid="stTable"] th:contains("description"),
    div[data-testid="stTable"] td:contains("description"),
    div[data-testid="stTable"] th:contains("reason"),
    div[data-testid="stTable"] td:contains("reason"),
    div[data-testid="stTable"] th:contains("sentence"),
    div[data-testid="stTable"] td:contains("sentence") {
        min-width: 300px !important;
        max-width: 500px !important;
        white-space: normal !important;
    }
    
    /* Last column - often contains longer text */
    div[data-testid="stTable"] th:last-child,
    div[data-testid="stTable"] td:last-child {
        min-width: 250px !important;
        max-width: 400px !important;
        white-space: normal !important;
    }
    
    /* Table hover effects */
    div[data-testid="stTable"] tbody tr:hover {
        background-color: #f9fafb !important;
    }
    
    /* Hide Streamlit elements that create empty spaces */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    .element-container {
        margin-bottom: 0 !important;
    }
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    /* Hide empty divs and containers */
    div:empty {
        display: none !important;
    }
    .row-widget {
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for instructions if not present
if 'custom_instructions' not in st.session_state:
    from src.utils.configs import (
        DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS,
        DEFAULT_PAPER_META_INFO_INSTRUCTIONS,
        DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS,
        DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS
    )
    
    st.session_state.custom_instructions = {
        'paper_relevance': DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS,
        'paper_meta_info': DEFAULT_PAPER_META_INFO_INSTRUCTIONS,
        'samples_extraction': DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS,
        'variables_extraction': DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS
    }

# Initialize sidebar state
if 'info_sidebar_open' not in st.session_state:
    st.session_state.info_sidebar_open = False

# Function to toggle sidebar
def toggle_info_sidebar():
    st.session_state.info_sidebar_open = not st.session_state.info_sidebar_open

# Function to reset the application state
def reset_application():
    """Reset application to initial state"""
    custom_instructions_backup = st.session_state.custom_instructions.copy()
    for key in list(st.session_state.keys()):
        if key not in ['custom_instructions']:
            del st.session_state[key]
    st.session_state.custom_instructions = custom_instructions_backup
    st.rerun()

# Function to save instructions
def save_instruction(instruction_type, new_content):
    st.session_state.custom_instructions[instruction_type] = new_content
    st.success(f"✅ {instruction_type.replace('_', ' ').title()} instructions updated!")

# App header with toggle button
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("# Meta-Analysis Agent")
    st.markdown("""
    <p class="subtitle">
    The Meta-Analysis Agent is an Gen-AI powered research tool for meta analysis data coding designed for academic researchers. 
    The AI agents help researchers efficiently and accurately code the relevant data given a research question and variables of interests.
    </p>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='padding-top: 1rem;'></div>", unsafe_allow_html=True)
    toggle_button_text = "✕ Close Info" if st.session_state.info_sidebar_open else "📖 About Agent"
    if st.button(toggle_button_text, key="sidebar_toggle", help="Toggle information sidebar"):
        toggle_info_sidebar()
        st.rerun()

# Info Sidebar Content (conditionally rendered)
if st.session_state.info_sidebar_open:
    with st.sidebar:
        st.markdown("# 📖 About the Agent")
        
        if st.button("✕ Close", key="close_sidebar", help="Close sidebar"):
            st.session_state.info_sidebar_open = False
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🔄 Workflow Overview")
        st.markdown("""
        1. **PDF Processing**: Documents parsed using LLaMa Parse
        2. **Relevance Check**: AI determines paper relevance
        3. **Data Extraction**: Structured extraction of metadata, samples, variables
        4. **Quality Control**: Confidence levels and reasons provided
        5. **Report Generation**: Comprehensive HTML reports created
        """)
        
        st.markdown("### ✨ Key Features")
        st.markdown("""
        - **🔍 Smart Variable Detection**: Finds variants and related measures
        - **📊 Correlation Extraction**: Identifies relationships between variables  
        - **⚖️ Quality Assessment**: Confidence levels for all extractions
        - **📋 Professional Reports**: Publication-ready HTML summaries
        - **🎯 Batch Processing**: Handle multiple papers simultaneously
        - **⚙️ Customizable Instructions**: Tailor AI behavior to your needs
        """)
        
        st.markdown("### 🔗 Extraction Pipeline")
        st.graphviz_chart("""
        digraph {
            rankdir=TB;
            node [shape=box, style=filled, fillcolor="#e0f2fe", fontsize=10];
            edge [color="#0284c7"];
            
            A [label="Load PDF", fillcolor="#bfdbfe"];
            B [label="Judge\\nRelevance", fillcolor="#bfdbfe"];
            C [label="Extract\\nMetadata", fillcolor="#93c5fd"];
            D [label="Extract\\nSamples", fillcolor="#93c5fd"];
            E [label="Extract\\nVariables", fillcolor="#60a5fa"];
            F [label="Generate\\nReport", fillcolor="#3b82f6", fontcolor="white"];
            
            A -> B;
            B -> C [label="✓"];
            B -> End [label="✗"];
            C -> D;
            D -> E;
            E -> F;
            F -> End [fillcolor="#1f2937", fontcolor="white"];
        }
        """)
        
        st.markdown("### 📈 Output Examples")
        st.markdown("""
        **CSV Files Generated:**
        - `papers.csv`: Metadata, confidence, reasons
        - `samples.csv`: Demographics, sample characteristics
        - `variables.csv`: Measures, reliability, statistics
        - `correlations.csv`: Correlation relationships (if selected)
        - `between_group_effects.csv`: Between-group comparisons (if selected)
        - `within_subject_effects.csv`: Within-subject/paired analyses (if selected)
        - `binary_event_effects.csv`: Binary outcomes and odds ratios (if selected)
        
        **HTML Reports:** Detailed analysis with researcher guidance
        
        **Effect-Size Types Available:**
        - 📊 Correlations (r) - Continuous variable relationships
        - ⚖️ Between-groups (Cohen's d) - Independent group comparisons  
        - 🔄 Within-subject (Cohen's d) - Paired/repeated measures
        - 🎯 Binary outcomes (OR) - Yes/no outcome analyses
        """)
        
        st.markdown("### 🎯 Best Practices")
        st.markdown("""
        - **Upload Quality**: Use clean, text-searchable PDFs
        - **Variable Definition**: Be specific and consistent
        - **Instruction Customization**: Tailor to your research domain
        - **Quality Review**: Always verify high-stakes extractions
        """)

# Main content area
if 'result_dir' in st.session_state:
    # Results view
    st.markdown('<div class="success-banner">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### ✅ Analysis Completed Successfully")
        st.markdown(f"**Results:** `{st.session_state.result_dir}`")
    with col2:
        if st.button("🔄 New Analysis", help="Start a new analysis", key="reset"):
            reset_application()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Results display
    st.markdown("## 📊 Analysis Results")
    
    # Fallback: Try to reload data if session state is missing dataframes
    if not hasattr(st.session_state, 'papers_df') and 'result_dir' in st.session_state:
        st.warning("🔄 Session state missing dataframes. Attempting to reload from files...")
        result_dir = st.session_state.result_dir
        try:
            # Reload core CSV files
            st.session_state.papers_df = pd.read_csv(os.path.join(result_dir, "papers.csv"))
            st.session_state.samples_df = pd.read_csv(os.path.join(result_dir, "samples.csv"))
            st.session_state.variables_df = pd.read_csv(os.path.join(result_dir, "variables.csv"))
            st.session_state.correlations_df = pd.read_csv(os.path.join(result_dir, "correlations.csv"))
            
            # Reload effect-specific files if they exist
            effect_files = {
                'between_group_effects.csv': 'between_group_effects_df',
                'within_subject_effects.csv': 'within_subject_effects_df',
                'binary_event_effects.csv': 'binary_event_effects_df'
            }
            
            for csv_file, attr_name in effect_files.items():
                csv_path = os.path.join(result_dir, csv_file)
                if os.path.exists(csv_path):
                    setattr(st.session_state, attr_name, pd.read_csv(csv_path))
            
            st.success("✅ Successfully reloaded data from files!")
            st.rerun()  # Refresh the UI
            
        except Exception as e:
            st.error(f"❌ Failed to reload data: {str(e)}")
    
    tabs = ["Papers", "Samples", "Variables", "Correlations"]
    
    # Add tabs for additional effect types if they exist
    if hasattr(st.session_state, 'between_group_effects_df'):
        tabs.append("Between-Group Effects")
    if hasattr(st.session_state, 'within_subject_effects_df'):
        tabs.append("Within-Subject Effects")  
    if hasattr(st.session_state, 'binary_event_effects_df'):
        tabs.append("Binary Event Effects")
    
    if 'html_reports' in st.session_state and st.session_state.html_reports:
        tabs.append("Reports")
    
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:  # Papers
        if hasattr(st.session_state, 'papers_df'):
            display_professional_table(st.session_state.papers_df, "Papers Analysis", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Papers Data",
                st.session_state.papers_df.to_csv(index=False).encode('utf-8'),
                "papers.csv", "text/csv", key='download-papers'
            )
        else:
            st.warning("📄 No papers data available. This could happen if:")
            st.markdown("- PDF files failed to parse")
            st.markdown("- All papers were marked as irrelevant")
            st.markdown("- Processing encountered errors")
    
    with tab_objects[1]:  # Samples
        if hasattr(st.session_state, 'samples_df'):
            display_professional_table(st.session_state.samples_df, "Sample Demographics", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Samples Data",
                st.session_state.samples_df.to_csv(index=False).encode('utf-8'),
                "samples.csv", "text/csv", key='download-samples'
            )
        else:
            st.warning("👥 No samples data available. This could happen if:")
            st.markdown("- No relevant papers were found")
            st.markdown("- Papers didn't contain sample information")
            st.markdown("- Sample extraction failed")
    
    with tab_objects[2]:  # Variables
        if hasattr(st.session_state, 'variables_df'):
            display_professional_table(st.session_state.variables_df, "Variable Information", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Variables Data",
                st.session_state.variables_df.to_csv(index=False).encode('utf-8'),
                "variables.csv", "text/csv", key='download-variables'
            )
        else:
            st.warning("🔬 No variables data available. This could happen if:")
            st.markdown("- No samples were extracted")
            st.markdown("- Papers didn't contain variable information")
            st.markdown("- Variable extraction failed")
    
    with tab_objects[3]:  # Correlations
        if hasattr(st.session_state, 'correlations_df'):
            display_professional_table(st.session_state.correlations_df, "Correlation Analysis", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Correlations Data",
                st.session_state.correlations_df.to_csv(index=False).encode('utf-8'),
                "correlations.csv", "text/csv", key='download-correlations'
            )
        else:
            st.warning("🔗 No correlations data available. This could happen if:")
            st.markdown("- No variables were extracted")
            st.markdown("- Papers didn't report correlations")
            st.markdown("- Correlation extraction failed")
    
    # Dynamic tabs for additional effect types
    current_tab_index = 4
    
    # Between-Group Effects Tab
    if hasattr(st.session_state, 'between_group_effects_df'):
        with tab_objects[current_tab_index]:
            df = st.session_state.between_group_effects_df
            display_professional_table(df, "Between-Group Effects", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Between-Group Effects Data",
                df.to_csv(index=False).encode('utf-8'),
                "between_group_effects.csv", "text/csv", key='download-between-group'
            )
        current_tab_index += 1
    
    # Within-Subject Effects Tab  
    if hasattr(st.session_state, 'within_subject_effects_df'):
        with tab_objects[current_tab_index]:
            df = st.session_state.within_subject_effects_df
            display_professional_table(df, "Within-Subject Effects", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Within-Subject Effects Data",
                df.to_csv(index=False).encode('utf-8'),
                "within_subject_effects.csv", "text/csv", key='download-within-subject'
            )
        current_tab_index += 1
    
    # Binary Event Effects Tab
    if hasattr(st.session_state, 'binary_event_effects_df'):
        with tab_objects[current_tab_index]:
            df = st.session_state.binary_event_effects_df
            display_professional_table(df, "Binary Event Effects", show_summary=True)
            
            # Download button
            st.download_button(
                "📥 Download Binary Event Effects Data",
                df.to_csv(index=False).encode('utf-8'),
                "binary_event_effects.csv", "text/csv", key='download-binary-event'
            )
        current_tab_index += 1
    
    # HTML Reports Tab
    if 'html_reports' in st.session_state and st.session_state.html_reports:
        with tab_objects[current_tab_index]:  # Reports
            st.markdown("### 📋 Extraction Reports")
            st.info("Detailed analysis reports with quality assessments and researcher guidance.")
            
            if st.session_state.html_reports:
                for html_file in st.session_state.html_reports:
                    filename = os.path.basename(html_file)
                    paper_id = filename.replace('_report.html', '')
                    
                    with st.expander(f"📄 {paper_id}", expanded=False):
                        try:
                            with open(html_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            st.components.v1.html(html_content, height=800, scrolling=True)
                            st.download_button(
                                f"📥 Download {filename}",
                                html_content, filename, "text/html",
                                key=f'download-html-{paper_id}'
                            )
                        except Exception as e:
                            st.error(f"Error loading report: {str(e)}")
    
    # Bulk downloads
    st.markdown("## 📦 Bulk Downloads")
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        # Always include these core files
        if hasattr(st.session_state, 'papers_df'):
            zip_file.writestr('papers.csv', st.session_state.papers_df.to_csv(index=False))
        if hasattr(st.session_state, 'samples_df'):
            zip_file.writestr('samples.csv', st.session_state.samples_df.to_csv(index=False))
        if hasattr(st.session_state, 'variables_df'):
            zip_file.writestr('variables.csv', st.session_state.variables_df.to_csv(index=False))
        if hasattr(st.session_state, 'correlations_df'):
            zip_file.writestr('correlations.csv', st.session_state.correlations_df.to_csv(index=False))
        
        # Include additional effect-specific files if they exist
        for effect_file in ['between_group_effects_df', 'within_subject_effects_df', 'binary_event_effects_df']:
            if hasattr(st.session_state, effect_file):
                csv_name = effect_file.replace('_df', '.csv')
                csv_data = getattr(st.session_state, effect_file).to_csv(index=False)
                zip_file.writestr(csv_name, csv_data)
        
        # Include HTML reports if available
        if 'html_reports' in st.session_state and st.session_state.html_reports:
            for html_file in st.session_state.html_reports:
                try:
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    zip_file.writestr(os.path.basename(html_file), html_content)
                except Exception as e:
                    logger.error(f"Error adding HTML file {html_file} to zip: {str(e)}")
    
    buffer.seek(0)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            label="📊 Download Complete Dataset",
            data=buffer.getvalue(),
            file_name="meta_analysis_results.zip",
            mime="application/zip",
            key='download-all'
        )
    with col2:
        # Count available files more accurately
        csv_files = sum([1 for attr in ['papers_df', 'samples_df', 'variables_df', 'correlations_df', 'between_group_effects_df', 'within_subject_effects_df', 'binary_event_effects_df'] if hasattr(st.session_state, attr)])
        html_files = len(st.session_state.get('html_reports', []))
        total_files = csv_files + html_files
        st.info(f"📁 Package contains {total_files} files")

else:
    # Initial workflow view - no extra containers here
    st.markdown('<div class="workflow-container">', unsafe_allow_html=True)
    
    # Step 1: Upload Papers
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">1</span>
        <div>
            <div class="step-title">Upload Academic Papers</div>
            <div class="step-description">Select PDF files for automated analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Select PDF files containing your academic papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF files for batch processing"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded: {', '.join([f.name for f in uploaded_files])}")
    else:
        st.info("📄 No files selected. Please upload PDF papers to begin.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 2: Define Variables
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">2</span>
        <div>
            <div class="step-title">Define Research Variables</div>
            <div class="step-description">Specify your dependent and independent variables</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        dependent_variable = st.text_input(
            "Dependent Variable",
            placeholder="e.g., workplace performance, academic achievement",
            help="The main outcome variable of interest"
        )
    
    with col2:
        independent_variables_input = st.text_input(
            "Independent Variables (comma-separated)",
            placeholder="e.g., leadership style, motivation, self-efficacy",
            help="Predictor variables that may influence the dependent variable"
        )
    
    # Effect-Size Type Selection
    st.markdown("**Effect-Size Type Selection**")
    effect_type_options = {
        "corr_r": "Correlation (r) between two continuous vars",
        "indep_d": "Independent-groups mean difference (Cohen's d)",
        "paired_d": "Within-subject / paired change (Cohen's d)",
        "binary_or": "Binary outcome — odds ratio (OR)"
    }
    
    effect_type_tooltips = {
        "corr_r": "Use when study reports r or stats to derive r.",
        "indep_d": "Two separate groups; need means & SDs or t/F.",
        "paired_d": "Same participants measured twice; need paired t or change scores.",
        "binary_or": "2×2 counts or reported OR for yes/no outcomes."
    }
    
    selected_effect_type = st.selectbox(
        "Select the type of effect size to extract:",
        options=list(effect_type_options.keys()),
        format_func=lambda x: effect_type_options[x],
        help="Choose the primary statistical relationship you want to extract from the papers",
        key="effect_type_selection"
    )
    
    # Show tooltip for selected effect type
    if selected_effect_type:
        st.info(f"💡 **{effect_type_options[selected_effect_type]}**: {effect_type_tooltips[selected_effect_type]}")
    
    # Conditional target groups input for independent groups
    target_groups = None
    if selected_effect_type == "indep_d":
        target_groups = st.text_input(
            "Target Groups for Comparison",
            placeholder="e.g., treatment vs control, high vs low dose",
            help="Specify which groups to compare (optional - if not specified, all group comparisons will be extracted)",
            key="target_groups_input"
        )
    
    # Show variable summary if both are provided
    if dependent_variable and independent_variables_input:
        independent_vars = [var.strip() for var in independent_variables_input.split(',')]
        effect_type_desc = effect_type_options[selected_effect_type]
        st.markdown(f"""
        <div class="variable-focus">
            🎯 <strong>Research Focus:</strong> {dependent_variable} ← {', '.join(independent_vars)}<br>
            📊 <strong>Effect Type:</strong> {effect_type_desc}
            {f'<br>👥 <strong>Target Groups:</strong> {target_groups}' if target_groups else ''}
        </div>
        """, unsafe_allow_html=True)
    elif dependent_variable or independent_variables_input:
        st.info("💡 Define both dependent and independent variables to proceed.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 3: Configure Instructions (Collapsible)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">3</span>
        <div>
            <div class="step-title">Configure Extraction Instructions</div>
            <div class="step-description">Optionally customize how the AI processes your papers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Customize AI Instructions", expanded=False):
        st.markdown("**Modify how the AI extracts and interprets data from your papers.**")
        
        instruction_tabs = st.tabs(["📋 Relevance", "📄 Metadata", "👥 Samples", "📊 Variables"])
        
        with instruction_tabs[0]:
            st.markdown("**Paper Relevance Assessment**")
            paper_relevance_new = st.text_area(
                "Instructions:", value=st.session_state.custom_instructions['paper_relevance'],
                height=120, key="paper_relevance_editor"
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 Save", key="save_paper_relevance"):
                    save_instruction('paper_relevance', paper_relevance_new)
            with col2:
                if st.button("🔄 Reset", key="reset_paper_relevance"):
                    from src.utils.configs import DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS
                    st.session_state.custom_instructions['paper_relevance'] = DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS
                    st.success("Reset to default!")
                    st.rerun()
        
        with instruction_tabs[1]:
            st.markdown("**Paper Metadata Extraction**")
            paper_meta_new = st.text_area(
                "Instructions:", value=st.session_state.custom_instructions['paper_meta_info'],
                height=120, key="paper_meta_editor"
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 Save", key="save_paper_meta"):
                    save_instruction('paper_meta_info', paper_meta_new)
            with col2:
                if st.button("🔄 Reset", key="reset_paper_meta"):
                    from src.utils.configs import DEFAULT_PAPER_META_INFO_INSTRUCTIONS
                    st.session_state.custom_instructions['paper_meta_info'] = DEFAULT_PAPER_META_INFO_INSTRUCTIONS
                    st.success("Reset to default!")
                    st.rerun()
        
        with instruction_tabs[2]:
            st.markdown("**Sample Information Extraction**")
            samples_new = st.text_area(
                "Instructions:", value=st.session_state.custom_instructions['samples_extraction'],
                height=120, key="samples_editor"
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 Save", key="save_samples"):
                    save_instruction('samples_extraction', samples_new)
            with col2:
                if st.button("🔄 Reset", key="reset_samples"):
                    from src.utils.configs import DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS
                    st.session_state.custom_instructions['samples_extraction'] = DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS
                    st.success("Reset to default!")
                    st.rerun()
        
        with instruction_tabs[3]:
            st.markdown("**Variable and Correlation Extraction**")
            variables_new = st.text_area(
                "Instructions:", value=st.session_state.custom_instructions['variables_extraction'],
                height=120, key="variables_editor"
            )
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("💾 Save", key="save_variables"):
                    save_instruction('variables_extraction', variables_new)
            with col2:
                if st.button("🔄 Reset", key="reset_variables"):
                    from src.utils.configs import DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS
                    st.session_state.custom_instructions['variables_extraction'] = DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS
                    st.success("Reset to default!")
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Start Analysis
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <span class="step-number">4</span>
        <div>
            <div class="step-title">Start Analysis</div>
            <div class="step-description">Launch the AI-powered extraction process</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Readiness check
    ready_to_process = uploaded_files and dependent_variable and independent_variables_input and selected_effect_type
    
    if not ready_to_process:
        st.markdown('<div class="requirements-list">', unsafe_allow_html=True)
        st.markdown("**Requirements Checklist:**")
        
        requirements = [
            ("📄 Papers uploaded", uploaded_files is not None),
            ("🎯 Dependent variable defined", bool(dependent_variable)),
            ("🔗 Independent variables defined", bool(independent_variables_input)),
            ("📊 Effect type selected", bool(selected_effect_type))
        ]
        
        for req_text, is_met in requirements:
            status = "✅" if is_met else "⏳"
            st.markdown(f"{status} {req_text}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    process_button = st.button(
        "🚀 Start Meta-Analysis Data Coding" if ready_to_process else "⏳ Complete Requirements Above",
        type="primary",
        disabled=not ready_to_process,
        use_container_width=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add extra space for progress display
    st.markdown('<div style="margin-bottom: 3rem;"></div>', unsafe_allow_html=True)
    
    # Process papers
    if uploaded_files and process_button and ready_to_process:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        independent_variables = [var.strip() for var in independent_variables_input.split(',')]
        
        from src.meta_agent.data_types import UserInstructions
        user_instructions = UserInstructions(
            paper_relevance_instructions=st.session_state.custom_instructions['paper_relevance'],
            paper_meta_info_instructions=st.session_state.custom_instructions['paper_meta_info'],
            samples_extraction_instructions=st.session_state.custom_instructions['samples_extraction'],
            variables_extraction_instructions=st.session_state.custom_instructions['variables_extraction']
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            status_text.text(f"📁 Preparing {len(uploaded_files)} files...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
            
            status_text.text("🤖 Running AI analysis...")
            try:
                result = process_papers(
                    temp_dir, 
                    "data/output", 
                    dependent_variable, 
                    independent_variables, 
                    user_instructions,
                    effect_types_to_extract=[selected_effect_type],  # Pass selected effect type as list
                    target_groups_for_comparison=target_groups  # Pass target groups if specified
                )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                timestamp_dirs = [d for d in os.listdir("data/output") if os.path.isdir(os.path.join("data/output", d))]
                if timestamp_dirs:
                    latest_dir = max(timestamp_dirs)
                    result_dir = os.path.join("data/output", latest_dir)
                    
                    st.session_state.result_dir = result_dir
                    
                    # Debug: Check if CSV files exist and log their sizes
                    csv_files = ["papers.csv", "samples.csv", "variables.csv", "correlations.csv"]
                    for csv_file in csv_files:
                        csv_path = os.path.join(result_dir, csv_file)
                        if os.path.exists(csv_path):
                            file_size = os.path.getsize(csv_path)
                            logger.info(f"Found {csv_file} - Size: {file_size} bytes")
                        else:
                            logger.error(f"Missing {csv_file} at {csv_path}")
                    
                    try:
                        # Load core CSV files with error handling
                        papers_csv_path = os.path.join(result_dir, "papers.csv")
                        samples_csv_path = os.path.join(result_dir, "samples.csv")
                        variables_csv_path = os.path.join(result_dir, "variables.csv")
                        correlations_csv_path = os.path.join(result_dir, "correlations.csv")
                        
                        st.session_state.papers_df = pd.read_csv(papers_csv_path)
                        st.session_state.samples_df = pd.read_csv(samples_csv_path)
                        st.session_state.variables_df = pd.read_csv(variables_csv_path)
                        st.session_state.correlations_df = pd.read_csv(correlations_csv_path)
                        
                        # Immediate verification after loading
                        logger.info(f"Immediate verification after loading:")
                        logger.info(f"  - papers_df in session_state: {hasattr(st.session_state, 'papers_df')}")
                        logger.info(f"  - papers_df shape: {st.session_state.papers_df.shape}")
                        logger.info(f"  - samples_df shape: {st.session_state.samples_df.shape}")
                        logger.info(f"  - variables_df shape: {st.session_state.variables_df.shape}")
                        logger.info(f"  - correlations_df shape: {st.session_state.correlations_df.shape}")
                        
                        # Debug: Log loaded dataframe info
                        logger.info(f"Successfully loaded core CSV files:")
                        logger.info(f"  - Papers: {len(st.session_state.papers_df)} rows, {len(st.session_state.papers_df.columns)} columns")
                        logger.info(f"  - Samples: {len(st.session_state.samples_df)} rows, {len(st.session_state.samples_df.columns)} columns") 
                        logger.info(f"  - Variables: {len(st.session_state.variables_df)} rows, {len(st.session_state.variables_df.columns)} columns")
                        logger.info(f"  - Correlations: {len(st.session_state.correlations_df)} rows, {len(st.session_state.correlations_df.columns)} columns")
                        
                        csv_load_success = True
                        
                    except Exception as e:
                        logger.error(f"Error loading core CSV files: {str(e)}")
                        st.error(f"Error loading CSV files: {str(e)}")
                        csv_load_success = False
                    
                    if csv_load_success:
                        # Load additional effect-specific CSV files if they exist
                        effect_files = {
                            'between_group_effects.csv': 'between_group_effects_df',
                            'within_subject_effects.csv': 'within_subject_effects_df',
                            'binary_event_effects.csv': 'binary_event_effects_df'
                        }
                        
                        for csv_file, attr_name in effect_files.items():
                            csv_path = os.path.join(result_dir, csv_file)
                            if os.path.exists(csv_path):
                                try:
                                    df = pd.read_csv(csv_path)
                                    setattr(st.session_state, attr_name, df)
                                    logger.info(f"Loaded {csv_file}: {len(df)} rows, {len(df.columns)} columns")
                                except Exception as e:
                                    logger.error(f"Error loading {csv_file}: {str(e)}")
                            else:
                                logger.info(f"{csv_file} not found - effect type not selected")
                        
                        # Debug: Verify session state attributes are set
                        session_attrs = ['papers_df', 'samples_df', 'variables_df', 'correlations_df', 
                                       'between_group_effects_df', 'within_subject_effects_df', 'binary_event_effects_df']
                        
                        logger.info("Session state verification:")
                        for attr in session_attrs:
                            if hasattr(st.session_state, attr):
                                df = getattr(st.session_state, attr)
                                logger.info(f"  ✅ {attr}: {len(df)} rows")
                            else:
                                logger.info(f"  ❌ {attr}: not set")
                        
                        html_files = glob.glob(os.path.join(result_dir, "*.html"))
                        st.session_state.html_reports = html_files
                        
                        # Show better success message with debug info
                        total_papers = len(st.session_state.papers_df)
                        if total_papers > 0:
                            st.success(f"🎉 Processing complete! Loaded {total_papers} papers with data.")
                        else:
                            st.warning("⚠️ Processing complete, but no paper data was extracted. Check the tabs for details.")
                        
                        # Set flag to indicate data is loaded successfully
                        st.session_state.data_loaded_successfully = True
                        
                        # Force a complete refresh
                        logger.info("Forcing app rerun after successful data loading")
                        st.rerun()
                else:
                    st.warning("⚠️ Processing completed but no results found.")
            
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                logger.exception("Error during processing")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>"
    "© 2025 Meta-Analysis Agent | Designed for Academic Research Excellence"
    "</div>", 
    unsafe_allow_html=True
) 