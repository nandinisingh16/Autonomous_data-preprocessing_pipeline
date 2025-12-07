
---


````md
# Quickstart (Deploy in 2 steps)

### Option A — Docker + n8n (Recommended)

```bash
docker-compose up -d
````

Open n8n:

> [http://localhost:5678](http://localhost:5678)
> Import the workflow file: `workflow_n8n.json`

Trigger any of these:

* File upload → auto process
* Webhook → ML-ready output

---

### Option B — Simple REST Endpoint

```bash
python simple_n8n_api.py
```

Run a dataset:

```bash
curl -X POST http://localhost:5000/webhook/start \
  -H "Content-Type: application/json" \
  -d '{"file_url":"<csv_url>","target_column":"Survived"}'
```

You will receive:
✔ cleaned dataset
✔ report + stats
✔ PTMA score

---

Next: [Product Overview »](product-overview.md)

