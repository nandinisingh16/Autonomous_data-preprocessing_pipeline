# test_metrics.py
from metrics_tracker import metrics

# Reset
metrics.reset()

# Simulate a pipeline run
metrics.task_executed()  # Ingestion
metrics.task_executed()  # Cleaning
metrics.auto_mod()       # Auto-cleaning
metrics.task_executed()  # Transformation
metrics.auto_mod()       # Auto-transformation
metrics.prompt_used()    # LLM call
metrics.task_executed()  # EDA
metrics.auto_mod()       # Auto-EDA
metrics.task_executed()  # Feature Engineering
metrics.human_mod()      # Human intervention
metrics.correction_made() # One correction needed

print("Test Metrics:")
print(metrics.to_dict())