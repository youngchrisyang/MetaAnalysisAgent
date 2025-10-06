## Configs used in the meta agent

DEFAULT_PAPER_RELEVANCE_INSTRUCTIONS = """
1. A paper is relevant if it contains empirical data studying the relationship between the specified variables, even if not the primary focus.
2. Include papers where the variables are measured as constructs, scales, or behavioral indicators - they don't need exact name matches.
3. Consider papers relevant if they study conceptually related constructs (e.g., "workplace aggression" for "hostility", "personality traits" for specific traits).
4. Include papers where both variables appear as independent variables, as predictors, or in correlation matrices.
5. Exclude purely theoretical papers, reviews without original data, meta-analyses, or studies lacking quantitative measures of the variables.
6. When in doubt, err on the side of inclusion - it's better to extract from a marginally relevant paper than miss a valuable one.
7. Look for the variables in the abstract, methods, results sections, and any tables or figures.
"""

DEFAULT_PAPER_META_INFO_INSTRUCTIONS = """
**Paper meta-information extraction**
- **Aggregating sub‑group statistics:**
  - When a paper reports descriptive statistics separately for sub‑groups (e.g., by age, gender or experimental condition), compute and report combined statistics for the full sample whenever possible.
  - Only keep sub‑group statistics if they are essential to the research question.
- **Multiple studies or waves:**
  - If a paper contains multiple studies, experiments, waves or time points and also reports combined data across them, treat the combined dataset as an additional "sample" in your coding alongside the individual study/wave samples.
- **Computing missing descriptive data:**
  - When demographic information (such as mean age) is not reported directly but can be calculated from available data (e.g., birth years or age distributions), compute and include it.
  - If information is not available, leave the field blank rather than guessing.
- **Sub‑group names in meta‑info:**
  - For meta‑information (sample description), do not differentiate between sub‑groups unless explicitly required by the analysis.  Describe the sample as a whole.
"""

DEFAULT_SAMPLES_EXTRACTION_INSTRUCTIONS = """
**Sample extraction**

1. **Identify and label distinct samples**
   - Count the number of distinct samples in the paper.  Distinct samples may arise from different studies, experiments, waves, conditions or time points.
   - Assign a concise name and description to each sample that captures the population and context (e.g., "online adult participants in Study 1" or "combined Study 1 + Study 2 sample").

2. **Extract standard sample characteristics**
   - Sample type (e.g., students, community volunteers, clinical participants, employees, incarcerated/offender group).
   - Sampling technique (e.g., convenience, random sampling, online recruitment).
   - Sample size, with counts by demographic groups where available.
   - Location (country or region).
   - Age statistics (mean and standard deviation) and other key demographics if reported or computable.
   - Any relevant notes about recruitment, ethnicity, response rate or other defining features.

3. **Handling sub‑groups and combined samples**
   - For descriptive purposes, merge sub‑group statistics into a single sample profile unless a specific sub‑group is of analytic interest.
   - If combined statistics across studies or waves are reported, treat the combined sample as a separate entry in addition to the individual samples.

4. **Missing information**
   - If specific sample information cannot be determined, leave the field blank.  Do not infer or fabricate data.
   - Do not calculate one information from another, such as subtracting male_n from total_n to get female_n, etc.
"""

DEFAULT_VARIABLES_EXTRACTION_INSTRUCTIONS = """
**Variable extraction**

1. **Identify variables and classify their roles**
   - List all variables of interest and distinguish between those acting as predictors/exposures (often called independent variables) and those acting as outcomes (dependent variables).
   - If the study uses conceptual categories (e.g., cognitive, behavioral, biological measures), note each variable's category where relevant.

2. **Sub‑scales and multiple methods**
   - Treat each sub‑dimension or sub‑scale of a measure as a separate variable.
   - When the same construct is measured using different instruments or methods (e.g., questionnaires versus tasks), code each method as a separate variable.
   - When a variable is measured at multiple time points or under different conditions, treat each measurement as a separate variable.

3. **Excluding control variables**
   - Do not code variables that are only used as control variables or covariates; focus on variables that are part of the core research questions.

4. **Link variables to samples**
   - For each variable, specify which sample(s) it pertains to.  If a variable is only measured in a specific study, time point or subgroup, link it accordingly.

5. **Composite measures and multiple reports**
   - When both sub‑scale scores and composite scores are available, code each.
   - If multiple instruments measure a similar construct, treat each instrument as a separate variable.

**Correlation and effect‑size extraction (optional)**

- **Extracting relationships:** For each sample, extract the reported correlations or other effect sizes between predictor and outcome variables.  Where multiple measures or time points exist, extract each relationship separately and label them clearly.
- **Combining sub‑group correlations:** If correlations are reported only by sub‑group and full‑sample values can be calculated from the available data, compute and report the full‑sample values.  Otherwise, note that correlations are sub‑group‑specific.
- **Orientation and consistency:** Ensure that the direction of relationships is consistent with your classification of predictor and outcome variables.  If a paper reports correlations in reversed order, adjust your coding accordingly
"""

# ========== SHARED QUALITY CONTROL PROMPTS ==========

CONFIDENCE_LEVEL_INSTRUCTIONS = """
**Confidence Assessment:**
For each piece of information you extract, assess your confidence level based on the clarity and availability of the source information:

- **"confident"**: 
The information is clearly stated in the paper, unambiguous, and directly extractable from explicit text or tables.
- **"with reservations"**: 
The information is present but requires some interpretation, calculation, or inference from the available data. 
There may be minor ambiguities but the extraction is reasonably well-supported. 
If the original paper included inconsistent information, such as the correlation reported in tables and in text are different, put “with reservation” and notify this in notes.
- **"not confident"**: The information is unclear, ambiguous, requires significant inference, or is not explicitly stated. The extraction may be based on limited evidence or assumptions.

Provide a single confidence level that reflects your overall confidence in the accuracy and completeness of your extraction for this item.
"""

WARNINGS_CALLOUTS_INSTRUCTIONS = """
**Warnings and Callouts:**
Identify specific issues that researchers should be aware of when interpreting this extracted information. Provide a list of warnings/callouts for any of the following situations:

- **Missing information**: Key details not reported (e.g., "Sample size not clearly stated", "Reliability not reported")
- **Ambiguous information**: Information that could be interpreted multiple ways (e.g., "Unclear if statistics are for full sample or subgroup")
- **Calculated/inferred data**: Information that required computation or inference (e.g., "Mean age calculated from age ranges", "Sample size inferred from degrees of freedom")
- **Methodological concerns**: Issues with measurement or reporting (e.g., "Non-standard scale used", "Correlation matrix incomplete")
- **Multiple interpretations**: Cases where the paper could support different extractions (e.g., "Variable name appears in multiple contexts", "Conflicting statistics reported")
- **Quality concerns**: Issues that might affect data quality (e.g., "Very low response rate", "Small sample size", "Outdated measurement instrument")

If no significant issues are identified, return an empty list. Focus on actionable concerns that would help researchers evaluate the reliability and usability of the extracted data.
"""