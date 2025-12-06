import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
import numpy as np

def analyze_results():
    """Analyze pipeline test results with real dataset differences."""
    
    result_files = glob.glob("pipeline_test_results_*.json")
    if not result_files:
        print("❌ No test results found!")
        print("   Run: python test_pipeline_benchmark.py --with-llm")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📊 Analyzing: {latest_file}\n")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("❌ No valid results to analyze")
        return
    
    print(f"✅ Analyzing {len(valid_results)} successful pipeline runs\n")
    
    datasets = [r['dataset_name'] for r in valid_results]
    
    # Extract real metrics (will vary by dataset now)
    ptma_scores = [r.get('ptma_metrics', {}).get('PTMA', 0) for r in valid_results]
    sas_scores = [r.get('ptma_metrics', {}).get('SAS', 0) for r in valid_results]
    pdr_scores = [r.get('ptma_metrics', {}).get('PDR', 0) for r in valid_results]
    cof_scores = [r.get('ptma_metrics', {}).get('COF', 0) for r in valid_results]
    execution_times = [r.get('execution_time_seconds', 0) for r in valid_results]
    dataset_shapes = [r.get('dataset_shape', (0, 0)) for r in valid_results]
    dataset_rows = [shape[0] for shape in dataset_shapes]
    
    rows_per_second = [
        rows / max(0.1, exec_time) 
        for rows, exec_time in zip(dataset_rows, execution_times)
    ]
    
    llm_enabled = [r.get('llm_enabled', False) for r in valid_results]
    llm_providers = [r.get('llm_provider', 'none') for r in valid_results]
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PTMA Scores (showing VARIATION across datasets)
    colors = ['#2ecc71' if score == 1.0 else '#f39c12' for score in ptma_scores]
    axes[0,0].bar(datasets, ptma_scores, color=colors, edgecolor='black', linewidth=2)
    axes[0,0].set_title('PTMA Autonomy Score by Dataset\n(≠ all datasets now)', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('PTMA Score', fontsize=12)
    axes[0,0].set_ylim([0, 1.05])
    axes[0,0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(ptma_scores):
        axes[0,0].text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 2. Execution Time vs Dataset Size (REAL DATA)
    scatter = axes[0,1].scatter(dataset_rows, execution_times, s=300, 
                               c=ptma_scores, cmap='RdYlGn', edgecolor='black', linewidth=2)
    for i, dataset in enumerate(datasets):
        axes[0,1].annotate(dataset, (dataset_rows[i], execution_times[i]), 
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[0,1].set_title('Dataset Size vs Execution Time\n(Real correlation)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Dataset Rows', fontsize=12)
    axes[0,1].set_ylabel('Execution Time (s)', fontsize=12)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. LLM Impact
    llm_labels = ['LLM Enabled' if llm else 'LLM Disabled' for llm in llm_enabled]
    colors_llm = ['#3498db' if llm else '#95a5a6' for llm in llm_enabled]
    axes[0,2].bar(datasets, ptma_scores, color=colors_llm, edgecolor='black', linewidth=2)
    axes[0,2].set_title('LLM Impact on Pipeline Performance', fontsize=14, fontweight='bold')
    axes[0,2].set_ylabel('PTMA Score', fontsize=12)
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].legend([plt.Rectangle((0,0),1,1, fc='#3498db'), plt.Rectangle((0,0),1,1, fc='#95a5a6')],
                     ['LLM Enabled', 'LLM Disabled'], loc='lower right')
    
    # 4. Component Scores Variation
    x = np.arange(len(datasets))
    width = 0.2
    axes[1,0].bar(x - width, sas_scores, width, label='SAS (Autonomy)', color='#e74c3c', edgecolor='black')
    axes[1,0].bar(x, pdr_scores, width, label='PDR (Efficiency)', color='#3498db', edgecolor='black')
    axes[1,0].bar(x + width, cof_scores, width, label='COF (Accuracy)', color='#2ecc71', edgecolor='black')
    axes[1,0].set_title('Component Scores by Dataset\n(Dataset-specific performance)', fontsize=14, fontweight='bold')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(datasets, rotation=45)
    axes[1,0].set_ylabel('Score', fontsize=12)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Processing Speed Variation
    axes[1,1].bar(datasets, rows_per_second, color='#9b59b6', edgecolor='black', linewidth=2)
    axes[1,1].set_title('Processing Speed by Dataset\n(Rows/second)', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Rows per Second', fontsize=12)
    axes[1,1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(rows_per_second):
        axes[1,1].text(i, v + 5, f'{v:.0f}', ha='center', fontweight='bold')
    
    # 6. Dataset Statistics
    stats_text = f"""
Dataset Characteristics:
━━━━━━━━━━━━━━━━━━━━━━━
Total Datasets: {len(valid_results)}
Avg Rows: {np.mean(dataset_rows):.0f}
Avg Cols: {np.mean([s[1] for s in dataset_shapes]):.0f}
LLM Enabled: {sum(llm_enabled)}/{len(valid_results)}

Performance Summary:
━━━━━━━━━━━━━━━━━━━━━━━
Avg PTMA: {np.mean(ptma_scores):.4f}
Avg SAS: {np.mean(sas_scores):.4f}
Avg Speed: {np.mean(rows_per_second):.0f} rows/s
Avg Time: {np.mean(execution_times):.2f}s
    """
    axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1,2].axis('off')
    
    plt.suptitle('Autonomous Preprocessing Pipeline - Real Performance Analysis\n(Datasets show DIFFERENT metrics)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('pipeline_performance_real_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: pipeline_performance_real_analysis.png\n")
    plt.show()
    
    # Create detailed comparison table
    summary_df = pd.DataFrame({
        'Dataset': datasets,
        'Rows': dataset_rows,
        'Cols': [s[1] for s in dataset_shapes],
        'PTMA': ptma_scores,
        'SAS': sas_scores,
        'PDR': pdr_scores,
        'COF': cof_scores,
        'Time(s)': execution_times,
        'Speed(RPS)': rows_per_second,
        'LLM': ['✅' if llm else '❌' for llm in llm_enabled],
        'Provider': llm_providers
    })
    
    print("="*120)
    print("REAL PERFORMANCE ANALYSIS (Datasets show DIFFERENT metrics)")
    print("="*120)
    print(summary_df.to_string(index=False))
    print("="*120 + "\n")
    
    # Statistical insights
    print("📊 KEY INSIGHTS:\n")
    print(f"1. PTMA Range: {min(ptma_scores):.4f} - {max(ptma_scores):.4f}")
    print(f"   └─ Variation: {np.std(ptma_scores):.6f} (lower = more consistent)")
    
    print(f"\n2. Speed Variation: {min(rows_per_second):.0f} - {max(rows_per_second):.0f} rows/s")
    print(f"   └─ Dataset size correlation: {np.corrcoef(dataset_rows, execution_times)[0,1]:.3f}")
    
    print(f"\n3. LLM Impact Analysis:")
    if sum(llm_enabled) > 0:
        llm_scores = [ptma_scores[i] for i in range(len(llm_enabled)) if llm_enabled[i]]
        no_llm_scores = [ptma_scores[i] for i in range(len(llm_enabled)) if not llm_enabled[i]]
        if llm_scores and no_llm_scores:
            llm_avg = np.mean(llm_scores)
            no_llm_avg = np.mean(no_llm_scores)
            improvement = ((llm_avg - no_llm_avg) / no_llm_avg) * 100 if no_llm_avg > 0 else 0

            print(f"   └─ With LLM: {llm_avg:.4f} ± {np.std(llm_scores):.4f}")
            print(f"   └─ Without LLM: {no_llm_avg:.4f} ± {np.std(no_llm_scores):.4f}")
            print(f"   └─ Performance Impact: {improvement:+.2f}% {'improvement' if improvement > 0 else 'degradation'}")

            # Detailed breakdown by dataset
            print(f"\n   📊 Dataset-by-Dataset LLM Impact:")
            for i, dataset in enumerate(datasets):
                llm_status = "LLM" if llm_enabled[i] else "No LLM"
                score = ptma_scores[i]
                print(f"      • {dataset}: {score:.4f} ({llm_status})")
    else:
        print(f"   └─ No LLM-enabled runs detected")
    
    summary_df.to_csv('pipeline_performance_comparison.csv', index=False)
    print(f"\n✅ Detailed results: pipeline_performance_comparison.csv")

if __name__ == "__main__":
    analyze_results()