

````md
# API Guide

## Endpoint

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/webhook/start` | Start full autonomous preprocessing |

---

### Request Payload

```json
{
  "file_url": "<public_csv_url>",
  "target_column": "Survived"
}
````

Optional:

* `"dataset_name": "Titanic"`
* `"auto_feature_engineer": true`

---

### Response Output

```json
{
  "status": "completed",
  "clean_file": "<url>",
  "report": "<url>",
  "ptma": 0.92,
  "time_sec": 24.1
}
```

Logs stream real-time execution.

---

### Errors & Recovery

Agents auto-retry:

* Missing headers → inferred
* Wrong types → corrected
* Outliers → flagged & fixed

---

Next: [n8n Integration »](n8n-integration.md)

````


