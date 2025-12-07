

---

#  Autonomous Agentic AI Data Preprocessing Pipeline

**A fully autonomous, multi-agent AI system for data preprocessing with n8n orchestration, API endpoints, real-time monitoring and research metrics proving autonomy**

[![Docs](https://img.shields.io/badge/Live%20Docs-GitHub%20Pages-brightgreen)](https://nandinisingh16.github.io/Autonomous-Agentic-Pipeline/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)]()
[![n8n](https://img.shields.io/badge/n8n-Orchestrated-orange)]()

> **No rules. No templates.**
> This pipeline decides *how* to clean your data — you just send the dataset.

---

## Project Highlight

| KPI                | Traditional | This System                       |
| ------------------ | ----------- | --------------------------------- |
| Manual effort      | High        | **↓ 88%**                         |
| Average runtime    | Minutes     | **20–30 sec**                     |
| Supervision needed | Always      | **0 corrections on 50% datasets** |
| Autonomy           | None        | **PTMA = 1.0 (Fully autonomous)** |

 Cuts delivery time
 Standardizes preprocessing quality
 Deployable instantly in existing workflows

---

##  What Makes It Product-Ready?

| Feature                       | Why it matters                            |
| ----------------------------- | ----------------------------------------- |
| Autonomous multi-agent design | Adjusts to any dataset — no config        |
| Self-correction & validation  | Reliable like a human analyst             |
| n8n workflow automation       | Plug into real business pipelines         |
| REST APIs                     | Upload file → get clean data              |
| Measurable autonomy (PTMA)    | Track AI performance like an employee KPI |

This isn’t a script —
**it’s an AI teammate for data teams.**

---

##  Architecture at a Glance

```
Webhook/API → Ingestion → Quality Check → EDA → Preprocessing
        ↓                                     ↓
  Metrics + Logs                    ML-ready Cleaned Data
```

**Specialized autonomous agents**
Ingestion | Cleaning | Validation | Feature Engineering | EDA | Split | Vectorization
All guided by **schema-aware planner + reward feedback**

---
---
<video src="../13.mp4" autoplay loop muted playsinline width="720"></video>



<video src="../16.mp4" controls loop muted playsinline width="720"></video>

<video src="../19.mp4" controls loop muted playsinline width="720"></video>
---

##  Results (4 Benchmark Datasets)

**Real autonomous behavior → real value**

| Metric               | Result                     |
| -------------------- | -------------------------- |
| Best PTMA            | **1.0 (fully autonomous)** |
| Avg Runtime          | **20–30 sec**              |
| Zero-error pipelines | **2/4 datasets**           |

<img width="829" height="443" src="https://github.com/user-attachments/assets/e2c65799-1635-4f83-a760-c1211d4f7045" />

> **Autonomy = measurable ROI**
> PTMA = SAS / (1 + PDR + COF)

---

###  Self-Audit & EDA (Automatic Reporting)

Readable by both ML engineers **and** downstream LLMs.
Used for smart decision-making.

<div style="display:flex; gap:8px;">
<img width="320" src="https://github.com/user-attachments/assets/7293a2cb-e1c8-4591-8c3f-1a08b6ccde93" />
<img width="320" src="https://github.com/user-attachments/assets/b4a6e829-e607-4070-8a2a-1ec265ba1533" />
</div>

---

### 🔍 Pipeline Execution Log — Proof of Autonomy

**7 autonomous tasks. 0 corrections. PTMA = 1.0**

<img width="898" src="https://github.com/user-attachments/assets/2c8bc02a-6a3c-4a39-9a50-d312d3585e1f" />

---

##  Deploy in Production

### Option A — n8n + Docker

```bash
docker-compose up -d
# n8n UI → http://localhost:5678
# Import workflow_n8n.json only once
```

### Option B — REST Webhook

```bash
curl -X POST http://localhost:5000/webhook/start \
 -H "Content-Type: application/json" \
 -d '{"file_url":"<csv_url>","target_column":"Survived"}'
```

### Option C — Local Demo

```bash
git clone https://github.com/nandinisingh16/Autonomous-Agentic-Pipeline.git
cd Autonomous-Agentic-Pipeline
pip install -r requirements.txt
jupyter notebook notebooks/run_pipeline_demo.ipynb
```

Tested on **Windows & Ubuntu**

---

##  Project Structure

```
agents/          → autonomous modules
orchestrator/    → planning + reward feedback
api/             → n8n + REST integration layer
docs/            → Product documentation site
tests/           → stress tests & edge cases
notebooks/       → business demo
```

---

##  Why It’s Different

| Existing Solutions        | This System                       |
| ------------------------- | --------------------------------- |
| Rule-based cleaning       | Learns decisions from data        |
| Fragile to schema changes | Self-adapting                     |
| One-engine EDA            | Multi-engine cross validation     |
| No autonomy tracking      | **PTMA = performance KPI for AI** |

This brings **real autonomy** into data engineering pipelines.

---

##  Tech Stack

**Python + Docker + n8n + Flask + Pandas + Scikit-Learn
YData Profiling + AutoViz + DataPrep**

Cloud-agnostic: works with **S3, GCS, Azure, APIs or local files**

---

##  Documentation

Full  docs + API reference available here:
👉 [https://nandinisingh16.github.io/Autonomous-Agentic-Pipeline/](https://nandinisingh16.github.io/Autonomous-Agentic-Pipeline/)

---

##  Research Track 

Novel **PTMA metric (Springer under review)** → for teams measuring AI independence.



---

