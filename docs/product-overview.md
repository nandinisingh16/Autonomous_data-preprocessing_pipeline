

```md
# Product Overview

A **multi-agent AI system** where each preprocessing step is handled by a smart autonomous agent.

---

## Architecture

````

Webhook/API
↓
Ingestion → Quality → EDA → Cleaning → Transform → Feature Agent → Split
↓
ML-ready data + Metrics + PTMA

```

The **orchestrator** dynamically:
- picks which agents to run
- changes order if needed
- verifies + fixes mistakes

---

## Key Innovation

| Mechanism | Effect |
|----------|--------|
| Dynamic Agent Delegation | Best agent chosen per dataset |
| Schema-Driven Planning   | Zero configuration needed |
| Self-Correction Loop     | Detects & fixes its own errors |
| Cross-Agent Feedback     | Higher stability & quality |
| PTMA Metric              | Measures autonomy vs prompt-dependence |

Agents behave like a **team-mate**, not a script.

---

Next: [API Guide »](api-guide.md)
```

---
