# Autonomous_data-preprocessing_pipeline
Agentic AI data preprocessing pipeline with LLM guidance

## Run the Ingestion Module
> python ingestion.py

Later, we will add the option to pass the file in different ways. For now its hard-coded

## Run the Cleaning Module
> python cleaning.py <name_of_raw_file>

The ingestion step saves a raw data CSV file in the directory data/raw. Pick the file name from there and pass it here. The output of this stage is stored in data/cleaned
