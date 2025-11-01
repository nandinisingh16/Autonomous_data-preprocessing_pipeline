# pipeline.py
from pipeline_context import PipelineContext
from llm_agent import LLMAgent
from ingestion import IngestionModule

context = PipelineContext()
llm_agent = LLMAgent(provider="dummy")

ingestion = IngestionModule(context, llm_agent)
success = ingestion.run(file_path="sample_data.csv")

if success:
    print("\nPipeline Ingestion Output Preview:")
    print(context.raw_data.head())
else:
    print("\nIngestion failed.")
