from pipeline_context import PipelineContext
#from llm_agent import LLMAgent
from ingestion import IngestionModule

# Test: Titanic dataset. Remove later
import pandas as pd
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.to_csv("sample_data.csv", index=False) 
print("Saved sample_data.csv successfully!")


#Runner
context = PipelineContext(stage_name="ingestion")
#llm_agent = LLMAgent(provider="dummy")

ingestion = IngestionModule(context) #when LLM is available, pass here
success = ingestion.run(file_path="sample_data.csv")

if success:
    print("\nPipeline Ingestion Output Preview:")
    print(context.raw_data.head())
else:
    print("\nIngestion failed.")
