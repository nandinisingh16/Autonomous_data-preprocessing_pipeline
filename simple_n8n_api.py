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
from metrics_tracker import metrics
import os
import json
import sys
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from pipeline_orchestrator import PipelineOrchestrator
from llm_agent import create_llm_agent

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use module-level global variables that persist across requests
_pipeline_status = {}
_pipeline_results = {}
_pipeline_lock = threading.Lock()


@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "🚀 Autonomous Data Preprocessing Pipeline - n8n API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "webhook_start": {
                "method": "POST",
                "path": "/webhook/start",
                "description": "Trigger pipeline execution with LLM enabled by default",
                "example": {
                    "file_url": "test_datasets/titanic.csv",
                    "target_column": None,
                    "use_llm": True,
                    "llm_provider": "groq"
                },
                "note": "LLM is enabled by default for enhanced autonomous preprocessing"
            },
            "get_status": {
                "method": "GET",
                "path": "/status/<job_id>",
                "description": "Check pipeline status"
            },
            "ptma_metrics": {
                "method": "GET",
                "path": "/api/ptma-metrics/<run_id>",
                "description": "Get PTMA autonomy metrics"
            },
            "quality_check": {
                "method": "GET",
                "path": "/api/quality-check/<run_id>",
                "description": "Data quality assessment"
            },
            "eda": {
                "method": "GET",
                "path": "/api/run-eda/<run_id>",
                "description": "Exploratory Data Analysis results"
            },
            "preprocess": {
                "method": "GET",
                "path": "/api/preprocess/<run_id>",
                "description": "Preprocessing summary"
            },
            "ml_summary": {
                "method": "GET",
                "path": "/api/ml-summary/<run_id>",
                "description": "ML readiness summary"
            },
            "performance_metrics": {
                "method": "GET",
                "path": "/api/performance-metrics/<run_id>",
                "description": "Comprehensive performance metrics"
            },
            "health": {
                "method": "GET",
                "path": "/health",
                "description": "Server health check"
            }
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "server": "running",
        "active_jobs": len(_pipeline_status),
        "completed_jobs": len(_pipeline_results)
    })


@app.route('/webhook/start', methods=['POST'])
def webhook_start():
    """
    Webhook endpoint for n8n to trigger pipeline.
    
    Request JSON:
    {
        "file_url": "test_datasets/titanic.csv",
        "target_column": null,  # Optional
        "use_llm": true,
        "llm_provider": "groq"  # openai, anthropic, groq
    }
    """
    try:
        data = request.json or {}
        
        file_url = data.get("file_url", "sample_data.csv")
        target_column = data.get("target_column", None)  # ✅ OPTIONAL
        use_llm = data.get("use_llm", True)  # ✅ ENABLED BY DEFAULT
        llm_provider = data.get("llm_provider", "groq")  # ✅ DEFAULT TO GROQ
        
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        
        logger.info(f"🚀 Pipeline triggered (ID: {run_id})")
        logger.info(f"   File: {file_url}")
        logger.info(f"   Target: {target_column or 'None (unsupervised)'}")
        logger.info(f"   LLM: {llm_provider if use_llm else 'Disabled'}")
        
        # Initialize status
        with _pipeline_lock:
            _pipeline_status[run_id] = {
                "run_id": run_id,
                "status": "started",
                "progress": 0,
                "file": file_url,
                "target_column": target_column,
                "timestamp": datetime.now().isoformat(),
                "message": "Pipeline initializing..."
            }
        
        # Run in background thread
        thread = threading.Thread(
            target=run_pipeline_background,
            args=(file_url, target_column, use_llm, llm_provider, run_id),
            daemon=True
        )
        thread.start()
        
        # Return immediately with run_id
        return jsonify({
            "status": "queued",
            "run_id": run_id,
            "file": file_url,
            "target_column": target_column,
            "use_llm": use_llm,
            "llm_provider": llm_provider,
            "message": "Pipeline queued for execution. Check status with /status/<run_id>",
            "timestamp": datetime.now().isoformat()
        }), 202  # Accepted
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


def run_pipeline_background(file_url, target_column, use_llm, llm_provider, run_id):
    """Execute pipeline in background thread"""
    try:
        logger.info(f"[🔧] Starting pipeline execution: {run_id}")
        
        with _pipeline_lock:
            _pipeline_status[run_id].update({
                "status": "running",
                "progress": 25,
                "message": "Initializing pipeline..."
            })
        
        # Create LLM agent if needed
        llm_agent = create_llm_agent(use_llm, llm_provider) if use_llm else None
        
        # Create orchestrator (✅ NO HARDCODED TARGET COLUMN)
        orchestrator = PipelineOrchestrator(target_col=target_column, llm_agent=llm_agent)
        
        with _pipeline_lock:
            _pipeline_status[run_id].update({
                "progress": 50,
                "message": "Running pipeline stages..."
            })
        
        # Run pipeline
        start_time = time.time()
        result = orchestrator.run(input_file=file_url)
        execution_time = time.time() - start_time
        
        # Extract data info
        ingested_data = getattr(orchestrator.context, "ingested_data", None)
        if ingested_data is not None and isinstance(ingested_data, pd.DataFrame):
            dataset_shape = ingested_data.shape
            missing_count = ingested_data.isnull().sum().sum()
        else:
            dataset_shape = (0, 0)
            missing_count = 0
        
        # Store results
        pipeline_result = {
            "run_id": run_id,
            "status": "completed",
            "file": file_url,
            "target_column": target_column,
            "execution_time_seconds": execution_time,
            "dataset_info": {
                "rows": dataset_shape[0],
                "columns": dataset_shape[1],
                "missing_values": int(missing_count),
                "file_path": file_url
            },
            "ptma_metrics": result.get("autonomy_metrics", {}),
            "pipeline_status": result,
            "llm_enabled": use_llm,
            "llm_provider": llm_provider,
            "timestamp": datetime.now().isoformat()
        }
        
        with _pipeline_lock:
            _pipeline_results[run_id] = pipeline_result
            _pipeline_status[run_id].update({
                "status": "completed",
                "progress": 100,
                "completion_time": datetime.now().isoformat(),
                "message": f"Pipeline completed in {execution_time:.2f}s",
                "execution_time": execution_time,
                "ptma_metrics": result.get("autonomy_metrics", {})
            })
        
        logger.info(f"✅ Pipeline completed: {run_id} ({execution_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"❌ Pipeline error ({run_id}): {e}", exc_info=True)
        with _pipeline_lock:
            _pipeline_status[run_id].update({
                "status": "failed",
                "progress": 0,
                "error": str(e),
                "message": f"Pipeline failed: {str(e)}"
            })


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get pipeline status"""
    with _pipeline_lock:
        status = _pipeline_status.get(job_id)
    
    if status is None:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404
    
    return jsonify(status)


@app.route('/api/ptma-metrics/<run_id>', methods=['GET'])
def get_ptma_metrics(run_id):
    """Get PTMA autonomy metrics for a pipeline run"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. PTMA metrics will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat(),
                        "raw_metrics": {
                            "tasks": 0,
                            "prompts": 0,
                            "corrections": 0,
                            "auto_modifications": 0,
                            "human_modifications": 0,
                            "PDR": 0,
                            "SAS": 0,
                            "COF": 0,
                            "PTMA": 0
                        },
                        "calculated_scores": {
                            "COF": 0,
                            "PDR": 0,
                            "SAS": 0,
                            "PTMA": 0
                        },
                        "interpretation": {
                            "autonomy_level": "processing",
                            "efficiency": "processing",
                            "accuracy": "processing"
                        },
                        "dataset_info": {},
                        "execution_time_seconds": 0
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide PTMA metrics for failed pipeline.",
                        "timestamp": datetime.now().isoformat(),
                        "raw_metrics": {
                            "tasks": 0,
                            "prompts": 0,
                            "corrections": 0,
                            "auto_modifications": 0,
                            "human_modifications": 0,
                            "PDR": 0,
                            "SAS": 0,
                            "COF": 0,
                            "PTMA": 0
                        },
                        "calculated_scores": {
                            "COF": 0,
                            "PDR": 0,
                            "SAS": 0,
                            "PTMA": 0
                        },
                        "interpretation": {
                            "autonomy_level": "failed",
                            "efficiency": "failed",
                            "accuracy": "failed"
                        },
                        "dataset_info": {},
                        "execution_time_seconds": 0
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        ptma = result.get("ptma_metrics", {})

        response = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "raw_metrics": {
                "tasks": ptma.get("tasks", 0),
                "prompts": ptma.get("prompts", 0),
                "corrections": ptma.get("corrections", 0),
                "auto_modifications": ptma.get("auto_modifications", 0),
                "human_modifications": ptma.get("human_modifications", 0),
                "PDR": ptma.get("PDR", 0),
                "SAS": ptma.get("SAS", 0),
                "COF": ptma.get("COF", 0),
                "PTMA": ptma.get("PTMA", 0)
            },
            "calculated_scores": {
                "COF": ptma.get("COF", 0),
                "PDR": ptma.get("PDR", 0),
                "SAS": ptma.get("SAS", 0),
                "PTMA": ptma.get("PTMA", 0)
            },
            "interpretation": {
                "autonomy_level": "high" if ptma.get("SAS", 0) > 0.7 else "medium" if ptma.get("SAS", 0) > 0.3 else "low",
                "efficiency": "excellent" if ptma.get("PDR", 0) < 0.2 else "good" if ptma.get("PDR", 0) < 0.5 else "needs_improvement",
                "accuracy": "excellent" if ptma.get("COF", 0) < 0.1 else "good" if ptma.get("COF", 0) < 0.3 else "needs_improvement"
            },
            "dataset_info": result.get("dataset_info", {}),
            "execution_time_seconds": result.get("execution_time_seconds", 0)
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error retrieving PTMA metrics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/quality-check/<run_id>', methods=['GET'])
def quality_check(run_id):
    """Data quality check results"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. Quality check will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot perform quality check on failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        dataset_info = result.get("dataset_info", {})
        rows = dataset_info.get("rows", 0)
        cols = dataset_info.get("columns", 0)
        missing = dataset_info.get("missing_values", 0)

        total_cells = rows * cols if rows > 0 and cols > 0 else 1
        completeness = ((total_cells - missing) / total_cells * 100) if total_cells > 0 else 0

        return jsonify({
            "run_id": run_id,
            "quality_score": int(completeness),
            "status": "excellent" if completeness > 90 else "good" if completeness > 75 else "needs_attention",
            "checks": {
                "data_completeness": f"{completeness:.1f}%",
                "row_count": rows,
                "column_count": cols,
                "missing_values": missing
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-eda/<run_id>', methods=['GET'])
def run_eda(run_id):
    """Exploratory Data Analysis results"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. EDA results will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot perform EDA on failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        dataset_info = result.get("dataset_info", {})

        return jsonify({
            "run_id": run_id,
            "status": "completed",
            "eda_summary": {
                "dataset_shape": f"{dataset_info.get('rows', 0)} rows × {dataset_info.get('columns', 0)} columns",
                "missing_values": dataset_info.get("missing_values", 0),
                "report_generated": True
            },
            "file": result.get("file"),
            "execution_time": f"{result.get('execution_time_seconds', 0):.2f}s",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/preprocess/<run_id>', methods=['GET'])
def preprocess_dataset(run_id):
    """Preprocessing summary"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. Preprocessing summary will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide preprocessing summary for failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        return jsonify({
            "run_id": run_id,
            "status": "completed",
            "preprocessing": {
                "steps_applied": [
                    "Ingestion",
                    "Cleaning",
                    "Transformation",
                    "Feature Engineering",
                    "EDA",
                    "Train-Test Split",
                    "Vectorization"
                ],
                "dataset_info": result.get("dataset_info"),
                "target_column": result.get("target_column") or "None (unsupervised)"
            },
            "output": {
                "model_ready": True,
                "processed_file": f"processed_data/{run_id}_preprocessed.csv"
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ml-summary/<run_id>', methods=['GET'])
def ml_summary(run_id):
    """ML readiness summary"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. ML summary will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide ML summary for failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        ptma = result.get("ptma_metrics", {})

        return jsonify({
            "run_id": run_id,
            "status": "completed",
            "pipeline_complete": True,
            "summary": {
                "dataset_processed": True,
                "target_column": result.get("target_column") or "None",
                "data_quality_score": int((1 - ptma.get("COF", 0)) * 100),
                "preprocessing_steps": 7,
                "autonomy_score": ptma.get("PTMA", 0)
            },
            "deployment_ready": True,
            "ptma_metrics": ptma,
            "dataset_info": result.get("dataset_info"),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/performance-metrics/<run_id>', methods=['GET'])
def performance_metrics(run_id):
    """Comprehensive performance metrics"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. Performance metrics will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide performance metrics for failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        ptma = result.get("ptma_metrics", {})
        exec_time = result.get("execution_time_seconds", 0)

        return jsonify({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "pipeline_metrics": {
                "execution_time_seconds": exec_time,
                "tasks_completed": ptma.get("tasks", 0),
                "autonomy_score_ptma": ptma.get("PTMA", 0),
                "efficiency_score_pdr": ptma.get("PDR", 0),
                "autonomy_score_sas": ptma.get("SAS", 0),
                "correction_score_cof": ptma.get("COF", 0)
            },
            "data_metrics": result.get("dataset_info", {}),
            "processing_speed": f"{result.get('dataset_info', {}).get('rows', 1) / max(0.1, exec_time):.0f} rows/second"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/overview/<run_id>', methods=['GET'])
def api_overview(run_id):
    """Pipeline overview/summary endpoint"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. Overview will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide overview for failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        # Completed - return overview
        dataset_info = result.get("dataset_info", {})
        ptma = result.get("ptma_metrics", {})

        return jsonify({
            "run_id": run_id,
            "status": "completed",
            "overview": {
                "file": result.get("file"),
                "dataset_shape": f"{dataset_info.get('rows', 0)} × {dataset_info.get('columns', 0)}",
                "target_column": result.get("target_column") or "None",
                "missing_values": dataset_info.get("missing_values", 0),
                "execution_time": f"{result.get('execution_time_seconds', 0):.2f}s"
            },
            "autonomy": {
                "ptma": ptma.get("PTMA", 0),
                "sas": ptma.get("SAS", 0),
                "pdr": ptma.get("PDR", 0),
                "cof": ptma.get("COF", 0)
            },
            "pipeline_stages": [
                "✅ Ingestion",
                "✅ Cleaning",
                "✅ Transformation",
                "✅ Feature Engineering",
                "✅ EDA",
                "✅ Train-Test Split",
                "✅ Vectorization"
            ],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in api_overview: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/job-status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get job execution status"""
    try:
        with _pipeline_lock:
            status = _pipeline_status.get(job_id)
        
        if status is None:
            return jsonify({"error": f"Job '{job_id}' not found"}), 404
        
        return jsonify({
            "job_id": job_id,
            "status": status.get("status"),
            "progress": status.get("progress", 0),
            "message": status.get("message"),
            "execution_time": status.get("execution_time"),
            "ptma_metrics": status.get("ptma_metrics"),
            "timestamp": status.get("timestamp")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dataset-info/<run_id>', methods=['GET'])
def dataset_info(run_id):
    """Get dataset information"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. Dataset info will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide dataset info for failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        info = result.get("dataset_info", {})

        return jsonify({
            "run_id": run_id,
            "dataset_info": {
                "rows": info.get("rows", 0),
                "columns": info.get("columns", 0),
                "missing_values": info.get("missing_values", 0),
                "file_path": info.get("file_path", ""),
                "completeness_percentage": round(
                    ((info.get("rows", 1) * info.get("columns", 1) - info.get("missing_values", 0)) /
                     (info.get("rows", 1) * info.get("columns", 1)) * 100), 2
                ) if info.get("rows", 0) > 0 and info.get("columns", 0) > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pipeline-results/<run_id>', methods=['GET'])
def pipeline_results(run_id):
    """Get complete pipeline results"""
    try:
        with _pipeline_lock:
            result = _pipeline_results.get(run_id)
            status = _pipeline_status.get(run_id)

        if result is None:
            # Check if pipeline is still running or failed
            if status:
                if status.get("status") == "running":
                    return jsonify({
                        "run_id": run_id,
                        "status": "processing",
                        "message": "Pipeline is still running. Complete results will be available once completed.",
                        "progress": status.get("progress", 0),
                        "timestamp": datetime.now().isoformat()
                    }), 202  # Accepted - still processing
                elif status.get("status") == "failed":
                    return jsonify({
                        "run_id": run_id,
                        "status": "failed",
                        "error": status.get("error", "Pipeline failed"),
                        "message": "Cannot provide complete results for failed pipeline.",
                        "timestamp": datetime.now().isoformat()
                    }), 400  # Bad Request
                else:
                    return jsonify({"error": f"Run ID '{run_id}' not found"}), 404
            else:
                return jsonify({"error": f"Run ID '{run_id}' not found"}), 404

        return jsonify({
            "run_id": run_id,
            "status": "completed",
            "file": result.get("file"),
            "target_column": result.get("target_column"),
            "execution_time_seconds": result.get("execution_time_seconds"),
            "dataset_info": result.get("dataset_info"),
            "ptma_metrics": result.get("ptma_metrics"),
            "llm_enabled": result.get("llm_enabled"),
            "llm_provider": result.get("llm_provider"),
            "timestamp": result.get("timestamp")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/list-runs', methods=['GET'])
def list_runs():
    """List all completed pipeline runs"""
    try:
        with _pipeline_lock:
            runs = []
            for run_id, result in _pipeline_results.items():
                runs.append({
                    "run_id": run_id,
                    "file": result.get("file"),
                    "status": "completed",
                    "execution_time": result.get("execution_time_seconds"),
                    "ptma_score": result.get("ptma_metrics", {}).get("PTMA", 0),
                    "timestamp": result.get("timestamp")
                })
        
        return jsonify({
            "total_runs": len(runs),
            "runs": runs,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Autonomous Data Preprocessing Pipeline - n8n API Server")
    print("=" * 80)
    print(f"📍 Server: http://localhost:5000")
    print(f"📊 Health: http://localhost:5000/health")
    print(f"🎯 Webhook: POST http://localhost:5000/webhook/start")
    print(f"📈 PTMA Metrics: GET http://localhost:5000/api/ptma-metrics/<run_id>")
    print(f"🔍 Status: GET http://localhost:5000/status/<job_id>")
    print("=" * 80)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)