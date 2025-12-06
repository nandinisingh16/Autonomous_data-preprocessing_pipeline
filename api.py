"""
N8N Integration API for Autonomous Data Preprocessing Pipeline
Provides REST endpoints for n8n workflow orchestration
"""
from flask import Flask, request, jsonify
import threading
import uuid
import time
import logging
from datetime import datetime
import os
import sys
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from pipeline_orchestrator import PipelineOrchestrator
from llm_agent import create_llm_agent

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global state
_pipeline_status = {}
_pipeline_results = {}
_pipeline_lock = threading.Lock()

#############################################
# ROUTES
#############################################

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🚀 Autonomous Data Preprocessing Pipeline - n8n API",
        "version": "1.0",
        "status": "running",
        "docs": "/health"
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(_pipeline_status),
        "completed_jobs": len(_pipeline_results)
    })


@app.route('/webhook/start', methods=['POST'])
def webhook_start():
    try:
        data = request.json or {}
        run_id = str(uuid.uuid4())

        file_url = data.get("file_url", "test_datasets/titanic.csv")
        target_column = data.get("target_column", None)
        use_llm = data.get("use_llm", True)
        llm_provider = data.get("llm_provider", "groq")

        logger.info(f"🔥 Triggering pipeline :: {run_id}")

        with _pipeline_lock:
            _pipeline_status[run_id] = {
                "run_id": run_id,
                "status": "queued",
                "progress": 0,
                "timestamp": datetime.now().isoformat()
            }

        thread = threading.Thread(
            target=run_pipeline_background,
            args=(file_url, target_column, use_llm, llm_provider, run_id),
            daemon=True
        )
        thread.start()

        return jsonify({
            "message": "Pipeline queued",
            "run_id": run_id,
            "file": file_url,
            "use_llm": use_llm,
            "llm_provider": llm_provider
        }), 202

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/status/<run_id>', methods=['GET'])
def get_status(run_id):
    with _pipeline_lock:
        status = _pipeline_status.get(run_id)
    return jsonify(status) if status else (jsonify({"error": "Run not found"}), 404)


@app.route('/api/ptma-metrics/<run_id>', methods=['GET'])
def get_ptma_metrics(run_id):
    result, status = _get_result_status(run_id)
    if isinstance(result, tuple):
        return result  # Contains HTTP response

    ptma = result.get("ptma_metrics", {})

    return jsonify({
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "ptma_raw": ptma,
        "scores": {
            "autonomy": ptma.get("SAS", 0),
            "efficiency": ptma.get("PDR", 0),
            "accuracy": ptma.get("COF", 0),
            "overall_PTMA": ptma.get("PTMA", 0)
        },
        "dataset_info": result.get("dataset_info", {}),
        "execution_time_seconds": result.get("execution_time_seconds", 0)
    })


@app.route('/api/quality-check/<run_id>', methods=['GET'])
def quality_check(run_id):
    result, status = _get_result_status(run_id)
    if isinstance(result, tuple):
        return result

    info = result.get("dataset_info", {})
    rows = info.get("rows", 0)
    cols = info.get("columns", 0)
    missing = info.get("missing_values", 0)

    total = max(rows * cols, 1)
    completeness = round((total - missing) / total * 100, 1)

    return jsonify({
        "run_id": run_id,
        "quality_score": completeness,
        "missing_values": missing,
        "dataset_shape": f"{rows} x {cols}",
        "status": "excellent" if completeness > 90 else "good" if completeness > 75 else "poor"
    })


@app.route('/api/run-eda/<run_id>', methods=['GET'])
def run_eda(run_id):
    result, status = _get_result_status(run_id)
    if isinstance(result, tuple):
        return result

    info = result.get("dataset_info", {})
    return jsonify({
        "run_id": run_id,
        "eda_status": "generated",
        "dataset_shape": f"{info.get('rows',0)} rows × {info.get('columns',0)} columns",
        "missing_values": info.get("missing_values", 0)
    })


@app.route('/api/preprocess/<run_id>', methods=['GET'])
def preprocess_dataset(run_id):
    result, status = _get_result_status(run_id)
    if isinstance(result, tuple):
        return result

    return jsonify({
        "run_id": run_id,
        "steps": [
            "Ingestion", "Cleaning", "Transformation",
            "Feature Engineering", "EDA",
            "Train-Test Split", "Vectorization"
        ],
        "dataset_info": result.get("dataset_info", {}),
        "processed_file": f"processed_data/{run_id}_preprocessed.csv"
    })


@app.route('/api/ml-summary/<run_id>', methods=['GET'])
def ml_summary(run_id):
    result, status = _get_result_status(run_id)
    if isinstance(result, tuple):
        return result

    ptma = result.get("ptma_metrics", {})
    info = result.get("dataset_info", {})

    return jsonify({
        "run_id": run_id,
        "model_ready": True,
        "autonomy_score": ptma.get("PTMA", 0),
        "data_quality": max(0, round((1 - ptma.get("COF", 0)) * 100)),
        "dataset_info": info
    })


@app.route('/api/performance-metrics/<run_id>', methods=['GET'])
def performance_metrics(run_id):
    result, status = _get_result_status(run_id)
    if isinstance(result, tuple):
        return result

    exec_time = result.get("execution_time_seconds", 0)
    ptma = result.get("ptma_metrics", {})

    return jsonify({
        "run_id": run_id,
        "execution_time_seconds": exec_time,
        "performance": {
            "PTMA": ptma.get("PTMA", 0),
            "COF": ptma.get("COF", 0),
            "PDR": ptma.get("PDR", 0),
            "SAS": ptma.get("SAS", 0)
        }
    })

#############################################
# BACKGROUND WORKER
#############################################

def run_pipeline_background(file_url, target_column, use_llm, llm_provider, run_id):
    try:
        with _pipeline_lock:
            _pipeline_status[run_id].update({"status": "running", "progress": 20})

        llm_agent = create_llm_agent(use_llm, llm_provider) if use_llm else None
        orchestrator = PipelineOrchestrator(target_col=target_column, llm_agent=llm_agent)

        start = time.time()
        result = orchestrator.run(input_file=file_url)
        exec_time = time.time() - start

        df = getattr(orchestrator.context, "ingested_data", pd.DataFrame())
        rows, cols = df.shape if isinstance(df, pd.DataFrame) else (0, 0)
        missing = int(df.isnull().sum().sum()) if rows else 0

        pipeline_result = {
            "file": file_url,
            "target_column": target_column,
            "execution_time_seconds": exec_time,
            "dataset_info": {
                "rows": rows,
                "columns": cols,
                "missing_values": missing
            },
            "ptma_metrics": result.get("autonomy_metrics", {}),
            "llm_provider": llm_provider,
            "timestamp": datetime.now().isoformat()
        }

        with _pipeline_lock:
            _pipeline_results[run_id] = pipeline_result
            _pipeline_status[run_id].update({"status": "completed", "progress": 100})

        logger.info(f"✔ Pipeline completed {run_id} — {exec_time:.2f}s")

    except Exception as e:
        logger.error(f"Pipeline failed {run_id}: {e}", exc_info=True)
        with _pipeline_lock:
            _pipeline_status[run_id].update({"status": "failed", "error": str(e), "progress": 0})


#############################################
# HELPER
#############################################

def _get_result_status(run_id):
    with _pipeline_lock:
        result = _pipeline_results.get(run_id)
        status = _pipeline_status.get(run_id)

    if not status:
        return jsonify({"error": "Run not found"}), 404
    if status.get("status") == "running":
        return jsonify({"status": "processing", "progress": status.get("progress", 0)}), 202
    if status.get("status") == "failed":
        return jsonify({"status": "failed", "error": status.get("error")}), 400
    return result, status


#############################################
# STARTUP
#############################################

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
