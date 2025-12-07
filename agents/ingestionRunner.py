from orchestrator.pipeline_context import PipelineContext
#from llm_agent import LLMAgent
from ingestion import IngestionModule
from metrics_tracker import metrics

# Test: Titanic dataset. Remove later
import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
df = pd.read_csv(url)
df.to_csv("sample_data.csv", index=False) 
print("Saved sample_data.csv successfully!")


#Runner
context = PipelineContext(stage_name="ingestion")

ingestion = IngestionModule(context) #when LLM is available, pass here
success = ingestion.run(file_path="sample_data.csv")

if success:
    print("\nPipeline Ingestion Output Preview:")
    print(context.raw_data.head())
else:
    print("\nIngestion failed.")

def run(self, *args, **kwargs):
    # mark this runner as a pipeline task
    try:
        metrics.task_executed()
    except Exception:
        pass


    # try to infer and increment finer-grain counters from results/context
    try:
        results = getattr(self, "ingestion_results", {}) or getattr(self, "results", {}) or {}
        if results.get("used_agent") or results.get("prompted") or results.get("agent_consulted"):
            metrics.prompt_used()
        if results.get("auto_applied") or results.get("applied_automatically") or results.get("auto_fixed"):
            metrics.auto_mod()
        if results.get("human_override") or results.get("manual_fix") or results.get("human_adjusted"):
            metrics.human_mod()
        retries = int(results.get("retries", 0) or 0)
        for _ in range(max(0, retries)):
            metrics.correction_made()
    except Exception:
        pass
