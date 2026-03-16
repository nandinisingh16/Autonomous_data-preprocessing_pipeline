import json
from pathlib import Path
p = Path(__file__).parent / 'benchmark_selected_datasets_results.json'
data = json.loads(p.read_text())
for ds, modes in data.get('results', {}).items():
    print(ds, list(modes.keys()))
print('\nAggregated:')
for ds, modes in data.get('aggregated', {}).items():
    print(ds, list(modes.keys()))
