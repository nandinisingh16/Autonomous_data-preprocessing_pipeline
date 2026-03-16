import json
import copy
import importlib.util
from pathlib import Path


def load_sim_module():
    spec = importlib.util.spec_from_file_location(
        "simulate_llm_variations",
        Path("scripts/simulate_llm_variations.py").resolve(),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_with_llm_modes_added_and_ptma_diff():
    sim = load_sim_module()

    # Load a copy of the current results json
    p = Path("scripts/benchmark_selected_datasets_results.json").resolve()
    assert p.exists(), "benchmark results JSON not found"

    data = json.loads(p.read_text())
    data_copy = copy.deepcopy(data)

    # Run the simulation (in-memory) and recompute aggregated
    data_copy = sim.simulate_and_append(data_copy)
    data_copy = sim.recompute_aggregated(data_copy)

    agg = data_copy.get("aggregated", {})

    # Assert the new modes exist for at least one dataset
    assert any("with_llm_varied" in modes and "with_llm" in modes for modes in agg.values())

    # Assert that PTMA differs between with_llm_varied and with_llm for at least one dataset
    diff_found = False
    for ds, modes in agg.items():
        if "with_llm_varied" in modes and "with_llm" in modes:
            ptma_var = modes["with_llm_varied"]["avg_autonomy_metrics"].get("PTMA", 0)
            ptma_base = modes["with_llm"]["avg_autonomy_metrics"].get("PTMA", 0)
            if abs(ptma_var - ptma_base) > 0.01:
                diff_found = True
                break

    assert diff_found, "Expected PTMA to differ between 'with_llm_varied' and 'with_llm' for at least one dataset"
