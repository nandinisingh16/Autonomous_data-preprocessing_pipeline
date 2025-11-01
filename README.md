# Autonomous_data-preprocessing_pipeline
Agentic AI data preprocessing pipeline with LLM guidance

## Run the Ingestion Module
> python ingestion.py

Later, we will add the option to pass the file in different ways. For now its hard-coded

## Run the Cleaning Module
> python cleaning.py {name_of_raw_file} {method}

The methods we can use are:
- drop
- median
- mean

Mean is the default method, if no arg is passed, all missing numeric values are filled with the column mean.
Also, for now, for all text columns, if field value is missing, row is dropped. Later delegate this to the LLM, deciding which columns are useful, and whether to drop an entire record on that basis.

The ingestion step saves a raw data CSV file in the directory data/raw. Pick the file name from there and pass it here. The output of this stage is stored in data/cleaned
