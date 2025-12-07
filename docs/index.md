# Autonomous Agentic Data Preprocessing Pipeline
**Powered by LLMs, Python Agents & n8n Automation**

A fully automated data-preprocessing system that ingests any dataset and performs:
- ingestion
- cleaning
- transformations
- feature engineering
- EDA
- train/test split
- vectorization
- autonomy scoring (PTMA)

## 🚀 Demo Video
[Click to watch the demo](#)
(*Upload to YouTube → paste link here*)

---

# 🧠 Project Overview
This pipeline mimics how a human data scientist works — but fully autonomous.
It uses Python agents + LLM reasoning + n8n orchestration to process datasets end-to-end.

---

# 🔧 Architecture

![Pipeline Architecture](architecture.png)

System Components:

- **n8n workflow** → triggers preprocessing jobs
- **Flask API** → exposes pipeline endpoints
- **Python Agentic Core** → orchestrator + LLM agent + metrics
- **PTMA Framework** → calculates autonomy score (PDR, SAS, COF)

---

# 🎥 Screenshots

### Workflow Automation (n8n)
![Workflow](workflow.png)

### Agent Execution Logs
![Screenshot](screenshot-1.png)

### EDA Output
![Screenshot](screenshot-2.png)

---

# 📁 Sample Dataset
This project includes a sample dataset for demonstration:

- [sample-input.csv](sample-input.csv)
- [example-output.csv](sample-output.csv)

---

# 📓 Try the Pipeline Locally
Clone and run the pipeline notebook:

```bash
git clone https://github.com/nandinisingh16/Autonomous-Agentic-Pipeline.git
cd Autonomous-Agentic-Pipeline
pip install -r requirements.txt
jupyter notebook
```

Open:

```
notebooks/run_pipeline_demo.ipynb
```

The notebook demonstrates:

* loading a dataset
* running full preprocess pipeline
* generating PTMA metrics
* viewing outputs

---

# 📐 PTMA Autonomy Metrics

The system measures:

* **PDR** – Prompt Dependency Ratio
* **SAS** – Self Autonomy Score
* **COF** – Correction Overhead Factor
* **PTMA** – Overall autonomy score

Example output:

```json
{
  "tasks": 7,
  "prompts": 2,
  "corrections": 0,
  "PTMA": 0.76
}
```

---

# 🛠 Tools & Tech Used

* Python
* Flask
* Pandas / NumPy
* LLM APIs
* n8n Automation
* Docker
* GitHub Actions
* MATPLOTLIB / Seaborn

---

# 📡 Contact

**Raj Nandini Singh**
AI/ML Developer | Python | Data Science
[LinkedIn](https://linkedin.com/in/raj-nandini-singh)
[GitHub](https://github.com/nandinisingh16)
