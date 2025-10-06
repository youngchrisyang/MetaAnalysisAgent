import logging
from typing import Annotated, List, Optional, Dict
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from itertools import combinations
import operator

from .data_types import (
    SampleBasicInfo,
    SampleCompleteInfo, 
    UserInstructions,
    VariableInfoInSample,
    CorrelationInfoInSample,
    BetweenGroupEffectInSample,
    WithinSubjectEffectInSample,
    BinaryEventEffectInSample,
    GroupInfo
)

from .chains import (
    get_identify_related_variables_chain,
    get_identify_correlation_chunks_chain,
    get_extract_correlations_from_chunk_chain,
    get_extract_between_group_effects_chain,
    get_extract_within_subject_effects_chain,
    get_extract_binary_event_effects_chain
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SampleGraphState(TypedDict):
    sample_basic_info: SampleBasicInfo  # Basic information about the sample
    paper_content: List[Document]  # Content of the paper
    dependent_variable: str  # The dependent variable being studied
    independent_variables: List[str]  # The independent variables being studied
    effect_types_to_extract: List[str]  # Which effect types to extract (corr_r, indep_d, paired_d, binary_or)
    target_groups_for_comparison: Optional[str]  # Specific groups to compare for between-group analysis
    user_instructions: Optional[UserInstructions]  # User instructions for extraction
    sample_complete_info: Annotated[Optional[SampleCompleteInfo], operator.add] = None  # Complete sample information result
    extracted_variables: Annotated[Optional[List[VariableInfoInSample]], operator.add] = None  # Intermediate extracted variables
    extracted_correlations: Annotated[Optional[List[CorrelationInfoInSample]], operator.add] = None  # Intermediate extracted correlations
    extracted_between_group_effects: Annotated[Optional[List[BetweenGroupEffectInSample]], operator.add] = None  # Intermediate extracted between-group effects
    extracted_within_subject_effects: Annotated[Optional[List[WithinSubjectEffectInSample]], operator.add] = None  # Intermediate extracted within-subject effects
    extracted_binary_event_effects: Annotated[Optional[List[BinaryEventEffectInSample]], operator.add] = None  # Intermediate extracted binary event effects
    worklog: Annotated[str, operator.add] = ""  # Log of work done

class SampleGraph:
    def __init__(self, selected_model: str = "LLM_GPT41_FALLBACK"):
        self.selected_model = selected_model

    def get_model_instance(self):
        """Get the actual model instance based on selected_model string"""
        from src.utils.models import (
            LLM_GPT5_FALLBACK,
            LLM_GPT41_FALLBACK,
            LLM_CLAUDE4_FALLBACK,
            LLM_CLAUDE45_FALLBACK
        )

        model_mapping = {
            "LLM_GPT5_FALLBACK": LLM_GPT5_FALLBACK,
            "LLM_GPT41_FALLBACK": LLM_GPT41_FALLBACK,
            "LLM_CLAUDE4_FALLBACK": LLM_CLAUDE4_FALLBACK,
            "LLM_CLAUDE45_FALLBACK": LLM_CLAUDE45_FALLBACK
        }
        return model_mapping.get(self.selected_model, LLM_GPT41_FALLBACK)

    def initialize_sample_processing(self, state: SampleGraphState) -> SampleGraphState:
        """Initialize the sample processing with basic setup and logging"""
        
        sample_name = state["sample_basic_info"].sample_name
        worklog_entry = f"Initializing sample processing for: {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])
        
        # Log sample basic information
        sample_info = state["sample_basic_info"]
        logger.info(f"Processing sample: {sample_name}")
        logger.info(f"Sample description: {sample_info.sample_description}")
        logger.info(f"Sample size: {sample_info.sample_size}")
        
        return {
            "worklog": state["worklog"]
        }

    def extract_sample_variables(self, state: SampleGraphState) -> SampleGraphState:
        """
        Extract and identify all related variables in the sample.
        This is the first step of the variable extraction process.
        """

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        
        worklog_entry = f"Extracting variables for sample {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])

        sample_description = f"""
            Sample Name: {sample_name}
            Sample Description: {sample_basic_info.sample_description}
            Sample Size: {sample_basic_info.sample_size}
        """
        
        # Get variables extraction instructions
        user_instructions = state.get("user_instructions", UserInstructions())
        instructions = user_instructions.variables_extraction_instructions or UserInstructions().variables_extraction_instructions

        # ========== STEP 1: Identify all related variables ==========
        logger.info(f"Step 1: Identifying all related variables for sample {sample_name}")
        
        identify_variables_chain = get_identify_related_variables_chain(llm=self.get_model_instance())
        variables_output = identify_variables_chain({
            "paper_content": state["paper_content"],
            "sample_description": sample_description,
            "dependent_variable": state["dependent_variable"],
            "independent_variables": state["independent_variables"],
            "instructions": instructions
        })

        all_variables = variables_output.related_variables
        logger.info(f"Variable extraction complete: Found {len(all_variables)} related variables: {[var.variable_name for var in all_variables]}")

        # Store the extracted variables in state for the correlation extraction node
        return {
            "extracted_variables": all_variables,
            "worklog": state["worklog"]
        }

    def extract_sample_correlations(self, state: SampleGraphState) -> SampleGraphState:
        """
        Extract correlations using a 3-step approach:
        1. Identify text chunks containing correlation information
        2. Extract correlations from each chunk
        3. Aggregate results
        """

        # Early exit if correlation extraction is not selected
        effect_types = state["effect_types_to_extract"]
        if "corr_r" not in effect_types:
            logger.info("Skipping correlation extraction - not selected in effect_types_to_extract")
            return {
                "extracted_correlations": [],
                "worklog": state.get("worklog", "") + "Skipped correlation extraction (not selected)\n"
            }

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name

        worklog_entry = f"Extracting correlations for sample {sample_name} using 3-step approach"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])

        # Get the extracted variables from the previous step
        all_variables = state.get("extracted_variables", [])

        if not all_variables:
            logger.warning(f"No variables found for correlation extraction in sample {sample_name}")
            return {
                "extracted_correlations": [],
                "worklog": state["worklog"]
            }

        sample_description = f"""
            Sample Name: {sample_name}
            Sample Description: {sample_basic_info.sample_description}
            Sample Size: {sample_basic_info.sample_size}
        """

        # Format variables list for prompts
        variables_list = "\n".join([f"- {var.variable_name}" for var in all_variables])

        logger.info(f"Starting correlation extraction for {len(all_variables)} variables: {[var.variable_name for var in all_variables]}")

        # ========== STEP 1: Identify text chunks with correlation information ==========
        logger.info(f"Step 1: Identifying text chunks containing correlation information")

        identify_chunks_chain = get_identify_correlation_chunks_chain(llm=self.get_model_instance())
        chunks_output = identify_chunks_chain({
            "paper_content": state["paper_content"],
            "sample_description": sample_description,
            "variables_list": variables_list
        })

        correlation_chunks = chunks_output.correlation_chunks
        logger.info(f"Step 1 complete: Found {len(correlation_chunks)} text chunks containing correlation information")

        if not correlation_chunks:
            logger.info(f"No correlation chunks found for sample {sample_name}")
            return {
                "extracted_correlations": [],
                "worklog": state["worklog"]
            }

        # ========== STEP 2: Extract correlations from each chunk ==========
        logger.info(f"Step 2: Extracting correlations from {len(correlation_chunks)} chunks")

        all_correlations = []
        correlations_chain = get_extract_correlations_from_chunk_chain(llm=self.get_model_instance())

        for chunk_idx, chunk in enumerate(correlation_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(correlation_chunks)}")

            chunk_correlations_output = correlations_chain({
                "sample_description": sample_description,
                "variables_list": variables_list,
                "text_chunk": chunk
            })

            # Handle case where output might be None or missing correlations
            chunk_correlations = []
            if chunk_correlations_output and hasattr(chunk_correlations_output, 'correlations'):
                chunk_correlations = chunk_correlations_output.correlations or []

            chunk_found = len(chunk_correlations)
            logger.info(f"Chunk {chunk_idx + 1} complete: Found {chunk_found} correlations")

            # Add correlations from this chunk
            all_correlations.extend(chunk_correlations)

        # ========== STEP 3: Aggregate and deduplicate results ==========
        logger.info(f"Step 3: Aggregating correlation results")

        # Remove duplicates based on variable pairs (all correlations should exist now)
        seen_pairs = set()
        deduplicated_correlations = []

        for correlation in all_correlations:
            # Create a standardized pair key (sorted to handle both directions)
            pair_key = tuple(sorted([correlation.variable1, correlation.variable2]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                deduplicated_correlations.append(correlation)
            else:
                logger.info(f"Removed duplicate correlation: {correlation.variable1} vs {correlation.variable2}")

        logger.info(f"Correlation extraction complete: Found {len(deduplicated_correlations)} unique correlations from {len(correlation_chunks)} chunks")

        # Store the extracted correlations in state for the synthesize node
        return {
            "extracted_correlations": deduplicated_correlations,
            "worklog": state["worklog"]
        }

    def extract_between_group_effects(self, state: SampleGraphState) -> SampleGraphState:
        """
        Extract between-group effects from the sample.
        This extracts statistical comparisons between different groups or conditions.
        """

        # Early exit if between-group effects extraction is not selected
        effect_types = state["effect_types_to_extract"]  # FIXED: Direct access, no fallback to all types
        if "indep_d" not in effect_types:
            logger.info("Skipping between-group effects extraction - not selected in effect_types_to_extract")
            return {
                "extracted_between_group_effects": [],
                "worklog": state.get("worklog", "") + "Skipped between-group effects extraction (not selected)\n"
            }

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        
        worklog_entry = f"Extracting between-group effects for sample {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])

        # Get the extracted variables from the variable extraction step
        all_variables = state.get("extracted_variables", [])
        
        if not all_variables:
            logger.warning(f"No variables found for between-group effects extraction in sample {sample_name}")
            return {
                "extracted_between_group_effects": [],
                "worklog": state["worklog"]
            }

        sample_description = f"""
            Sample Name: {sample_name}
            Sample Description: {sample_basic_info.sample_description}
            Sample Size: {sample_basic_info.sample_size}
        """

        # Format variables list for the prompt
        variables_list = "\n".join([f"- {var.variable_name}: {var.scale_measure}" for var in all_variables])

        # Add target groups specification if provided
        target_groups = state.get("target_groups_for_comparison")
        if target_groups:
            sample_description += f"\n            Target Groups for Comparison: {target_groups}"
            logger.info(f"Using target groups specification: {target_groups}")

        logger.info(f"Analyzing group separations and extracting between-group effects for sample {sample_name}")
        
        # Use LLM chain to determine if group separations exist and extract effects
        between_group_chain = get_extract_between_group_effects_chain(llm=self.get_model_instance())
        between_group_output = between_group_chain({
            "paper_content": state["paper_content"],
            "sample_description": sample_description,
            "variables_list": variables_list
        })
        
        # Check LLM's judgment about group separations
        has_group_separations = between_group_output.has_group_separations
        all_between_group_effects = between_group_output.between_group_effects
        
        if not has_group_separations:
            logger.info(f"LLM determined no group separations exist in sample {sample_name}. Skipping extraction.")
            return {
                "extracted_between_group_effects": [],
                "worklog": state["worklog"] + f"Skipped between-group effects extraction - LLM determined no group separations exist\n"
            }
        
        logger.info(f"LLM determined group separations exist in sample {sample_name}. Found {len([e for e in all_between_group_effects if e.exists])} effects out of {len(all_between_group_effects)} checked")

        # Store the extracted between-group effects in state for the synthesize node
        return {
            "extracted_between_group_effects": all_between_group_effects,
            "worklog": state["worklog"]
        }

    def extract_within_subject_effects(self, state: SampleGraphState) -> SampleGraphState:
        """
        Extract within-subject effects from the sample.
        This extracts statistical comparisons within the same participants across conditions or time.
        """

        # Early exit if within-subject effects extraction is not selected
        effect_types = state["effect_types_to_extract"]  # FIXED: Direct access, no fallback to all types
        if "paired_d" not in effect_types:
            logger.info("Skipping within-subject effects extraction - not selected in effect_types_to_extract")
            return {
                "extracted_within_subject_effects": [],
                "worklog": state.get("worklog", "") + "Skipped within-subject effects extraction (not selected)\n"
            }

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        
        worklog_entry = f"Extracting within-subject effects for sample {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])

        # Get the extracted variables from the variable extraction step
        all_variables = state.get("extracted_variables", [])
        
        if not all_variables:
            logger.warning(f"No variables found for within-subject effects extraction in sample {sample_name}")
            return {
                "extracted_within_subject_effects": [],
                "worklog": state["worklog"]
            }

        sample_description = f"""
            Sample Name: {sample_name}
            Sample Description: {sample_basic_info.sample_description}
            Sample Size: {sample_basic_info.sample_size}
        """

        # Format variables list for the prompt
        variables_list = "\n".join([f"- {var.variable_name}: {var.scale_measure}" for var in all_variables])

        logger.info(f"Extracting within-subject effects for sample {sample_name}")
        
        # Extract within-subject effects
        within_subject_chain = get_extract_within_subject_effects_chain(llm=self.get_model_instance())
        within_subject_output = within_subject_chain({
            "paper_content": state["paper_content"],
            "sample_description": sample_description,
            "variables_list": variables_list
        })
        
        all_within_subject_effects = within_subject_output.within_subject_effects
        logger.info(f"Within-subject effects extraction complete: Found {len([e for e in all_within_subject_effects if e.exists])} effects out of {len(all_within_subject_effects)} checked")

        # Store the extracted within-subject effects in state for the synthesize node
        return {
            "extracted_within_subject_effects": all_within_subject_effects,
            "worklog": state["worklog"]
        }

    def extract_binary_event_effects(self, state: SampleGraphState) -> SampleGraphState:
        """
        Extract binary event effects from the sample.
        This extracts analyses involving binary outcomes and odds ratios.
        """

        # Early exit if binary event effects extraction is not selected
        effect_types = state["effect_types_to_extract"]  # FIXED: Direct access, no fallback to all types
        if "binary_or" not in effect_types:
            logger.info("Skipping binary event effects extraction - not selected in effect_types_to_extract")
            return {
                "extracted_binary_event_effects": [],
                "worklog": state.get("worklog", "") + "Skipped binary event effects extraction (not selected)\n"
            }

        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        
        worklog_entry = f"Extracting binary event effects for sample {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])

        # Get the extracted variables from the variable extraction step
        all_variables = state.get("extracted_variables", [])
        
        if not all_variables:
            logger.warning(f"No variables found for binary event effects extraction in sample {sample_name}")
            return {
                "extracted_binary_event_effects": [],
                "worklog": state["worklog"]
            }

        sample_description = f"""
            Sample Name: {sample_name}
            Sample Description: {sample_basic_info.sample_description}
            Sample Size: {sample_basic_info.sample_size}
        """

        # Format variables list for the prompt
        variables_list = "\n".join([f"- {var.variable_name}: {var.variable_type} - {var.scale_measure}" for var in all_variables])

        logger.info(f"Extracting binary event effects for sample {sample_name}")
        
        # Extract binary event effects
        binary_event_chain = get_extract_binary_event_effects_chain(llm=self.get_model_instance())
        binary_event_output = binary_event_chain({
            "paper_content": state["paper_content"],
            "sample_description": sample_description,
            "variables_list": variables_list
        })
        
        all_binary_event_effects = binary_event_output.binary_event_effects
        logger.info(f"Binary event effects extraction complete: Found {len([e for e in all_binary_event_effects if e.exists])} effects out of {len(all_binary_event_effects)} checked")

        # Store the extracted binary event effects in state for the synthesize node
        return {
            "extracted_binary_event_effects": all_binary_event_effects,
            "worklog": state["worklog"]
        }

    def synthesize_sample_info(self, state: SampleGraphState) -> SampleGraphState:
        """Synthesize the extracted information into a SampleCompleteInfo object"""
        
        sample_basic_info = state["sample_basic_info"]
        sample_name = sample_basic_info.sample_name
        
        worklog_entry = f"Synthesizing complete sample info for: {sample_name}"
        state["worklog"] = state.get("worklog", "") + worklog_entry + "\n"
        logger.info("Sample Graph Worklog: %s", state["worklog"])

        # Get all the extracted data from previous steps
        all_variables = state.get("extracted_variables", [])
        all_correlations = state.get("extracted_correlations", [])
        all_between_group_effects = state.get("extracted_between_group_effects", [])
        all_within_subject_effects = state.get("extracted_within_subject_effects", [])
        all_binary_event_effects = state.get("extracted_binary_event_effects", [])

        # Log which effect types were actually processed vs skipped
        effect_types = state["effect_types_to_extract"]  # FIXED: Direct access, no fallback to all types
        processed_types = []
        skipped_types = []
        
        effect_type_mapping = {
            "corr_r": ("correlations", len(all_correlations)),
            "indep_d": ("between-group effects", len(all_between_group_effects)),
            "paired_d": ("within-subject effects", len(all_within_subject_effects)),
            "binary_or": ("binary event effects", len(all_binary_event_effects))
        }
        
        for effect_type, (effect_name, count) in effect_type_mapping.items():
            if effect_type in effect_types:
                processed_types.append(f"{effect_name} ({count})")
            else:
                skipped_types.append(effect_name)
        
        logger.info(f"Effect extraction summary for {sample_name}:")
        logger.info(f"  ✅ Processed: {', '.join(processed_types) if processed_types else 'None'}")
        logger.info(f"  ⏭️  Skipped: {', '.join(skipped_types) if skipped_types else 'None'}")

        # Construct SampleCompleteInfo with all four types of effects
        sample_complete_info = SampleCompleteInfo(
            sample_name=sample_name,
            sample_basic_info=sample_basic_info,
            variables_info=all_variables,
            correlations_info=all_correlations,
            between_group_effects_info=all_between_group_effects,
            within_subject_effects_info=all_within_subject_effects,
            binary_event_effects_info=all_binary_event_effects
        )

        logger.info(f"Synthesize Complete: Constructed SampleCompleteInfo with:")
        logger.info(f"  - {len(all_variables)} variables")
        logger.info(f"  - {len(all_correlations)} correlation checks")
        logger.info(f"  - {len([e for e in all_between_group_effects if e.exists])} between-group effects")
        logger.info(f"  - {len([e for e in all_within_subject_effects if e.exists])} within-subject effects")
        logger.info(f"  - {len([e for e in all_binary_event_effects if e.exists])} binary event effects")

        return {
            "sample_complete_info": sample_complete_info,
            "worklog": state["worklog"]
        }

    def create_sample_graph(self):
        """Create the sample processing graph with conditional extraction nodes"""
        workflow = StateGraph(SampleGraphState)
        
        # Always add these core nodes
        workflow.add_node("initialize_sample_processing", self.initialize_sample_processing)
        workflow.add_node("extract_sample_variables", self.extract_sample_variables)
        workflow.add_node("synthesize_sample_info", self.synthesize_sample_info)
        
        # Add all extraction nodes - early exit logic in each node handles selection
        workflow.add_node("extract_sample_correlations", self.extract_sample_correlations)
        workflow.add_node("extract_between_group_effects", self.extract_between_group_effects)
        workflow.add_node("extract_within_subject_effects", self.extract_within_subject_effects)
        workflow.add_node("extract_binary_event_effects", self.extract_binary_event_effects)
        
        # Set entry point
        workflow.set_entry_point("initialize_sample_processing")
        
        # Add edges - simplified structure with early exit logic handling selection
        workflow.add_edge("initialize_sample_processing", "extract_sample_variables")
        
        # All extraction types run in parallel after variable extraction
        # Early exit logic in each node handles whether to actually process
        workflow.add_edge("extract_sample_variables", "extract_sample_correlations")
        workflow.add_edge("extract_sample_variables", "extract_between_group_effects")
        workflow.add_edge("extract_sample_variables", "extract_within_subject_effects")
        workflow.add_edge("extract_sample_variables", "extract_binary_event_effects")
        
        # All extraction nodes connect to synthesize
        workflow.add_edge("extract_sample_correlations", "synthesize_sample_info")
        workflow.add_edge("extract_between_group_effects", "synthesize_sample_info")
        workflow.add_edge("extract_within_subject_effects", "synthesize_sample_info")
        workflow.add_edge("extract_binary_event_effects", "synthesize_sample_info")
        
        workflow.add_edge("synthesize_sample_info", END)

        graph = workflow.compile()
        logger.info("Sample Graph ASCII:")
        logger.info(graph.get_graph().draw_ascii())
        return graph
