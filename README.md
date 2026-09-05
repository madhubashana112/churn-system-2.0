# Domain-Adaptive Churn Prediction Platform

A multi-tenant churn prediction platform that adapts to three industry verticals — SaaS,
Telecom/ISP and FinTech/Banking — with automated schema mapping, feature synthesis, risk
scoring and a sector-specific analytics dashboard per tenant.

Raw exports go in (CSV, TSV or a multi-sheet `.xlsx` workbook); a resolved schema, a scored
customer list and a retention playbook per customer come out.

## How it works

1. **Onboarding** — register a tenant and pick a sector. The sector chooses the retention core
   and the dashboard template; it is stored on the tenant, never taken from the browser.
2. **Upload** — drop in that sector's exports. The platform discovers the join key and each
   table's role (dimension, event log, transactional, free text) and discards noise columns.
3. **Feature synthesis** — rolling 7-day vs prior-30-day activity velocity, failure ratios,
   recency, keyword sentiment over free-text columns, plus sector math (balance drain and P2P
   failure streaks for FinTech, recharge gaps and tower concentration for Telecom, export
   share for SaaS). Windows are anchored to the newest timestamp in the upload, not to the
   wall clock, so the same files always produce the same features.
4. **Scoring** — the sector core scores every customer and emits a churn probability, a risk
   tier, sector-specific findings and a retention playbook.
5. **Dashboard** — sector KPI cards, charts, a focused intervention queue and a filterable
   customer table. Clicking a customer opens the evidence drawer: why they were flagged, the
   measured values behind it, and the recommended intervention.

## Offline mode (no API key required)

With no `DASHSCOPE_API_KEY` configured the platform selects a deterministic offline gateway.
It satisfies the same contract as the live Qwen client but computes every score from the
uploaded feature values, so the whole pipeline — including all three dashboards — runs with
no credentials and no network calls. Scores are a pure function of the input: the same upload
always yields the same numbers.

Two consequences worth knowing:

- The offline scorer normalises signals across the batch it is handed, so it is deliberately
  run in a single pass (`BATCH_SIZE=0`) rather than chunked. Slicing the population would
  re-normalise each slice against its own worst customer.
- The dashboard banner states plainly that scores were computed locally and are not Qwen
  output.

To score with the live model instead, put a key in `api_key.env` (or `.env`):

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

`QWEN_MODE=live` forces the live client even if key detection misfires; `QWEN_MODE=mock`
forces the offline gateway even when a key is present.

## Project structure

```text
├── churn_platform/
│   ├── application/           # Use cases and DTOs (no framework imports)
│   │   ├── dtos/              #   AnalysisResponse, MetricsSummary
│   │   └── use_cases/         #   schema resolution, synthesis, batching, summarising
│   ├── domain/                # Models and interfaces (no dependencies)
│   ├── infrastructure/
│   │   ├── ai/                #   Qwen gateway, offline gateway, sector cores, prompts
│   │   ├── parsers/           #   ingestion, feature synthesis, sentiment, enrichers
│   │   └── repositories/      #   in-memory tenant and analysis stores
│   ├── presentation/
│   │   ├── api/v1/            #   tenants, upload/analyze, analytics
│   │   ├── static/            #   app.js, styles.css
│   │   └── templates/         #   base + one dashboard per sector
│   ├── config.py              # Settings from environment / .env
│   └── main.py                # FastAPI app and sector routing
├── data/                      # Generated mock datasets (fintech, saas, telecom)
├── tests/                     # pytest suite, no network access
├── generate_mock_data.py      # Seeded mock dataset generator
├── pytest.ini                 # Test runner configuration
├── requirements.txt
└── api_key.env.example
```

## Quick start

### 1. Prerequisites

- Python 3.10+ (developed and tested on 3.14)
- Git

### 2. Set up a virtual environment

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration (optional)

```bash
cp api_key.env.example api_key.env
```

Leave the placeholder in place and the platform runs offline; replace it with a real key to
score with Qwen-Max.

### 5. Generate mock data (optional)

```bash
python generate_mock_data.py
```

Writes four CSVs per sector into `data/`. The generator is seeded: in every sector the first
25 entities carry deliberate churn patterns (collapsed usage, failed payments, dropped calls
on one tower, draining balances) and the remaining 75 are healthy, which is what makes the
dashboards show real separation instead of noise.

### 6. Run the application

```bash
uvicorn churn_platform.main:app --host 127.0.0.1 --port 8000
```

Then open:

- **Onboarding**: <http://127.0.0.1:8000/>
- **Dashboard**: <http://127.0.0.1:8000/dashboard> (redirects to your tenant's sector)
- **API documentation (Swagger UI)**: <http://127.0.0.1:8000/docs>

Upload the four CSVs of one sector from `data/<sector>/` to see that sector's dashboard
populated end to end.

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `DASHSCOPE_API_KEY` / `ALIBABA_API_KEY` | unset | Enables the live Qwen client |
| `QWEN_MODE` | `auto` | `auto` picks live when a key exists, else offline; `live` / `mock` force one |
| `QWEN_MODEL` | `qwen-max` | Model name passed to the live client |
| `QWEN_BASE_URL` | DashScope intl endpoint | OpenAI-compatible base URL |
| `BATCH_SIZE` | `10` | Customers per scoring call in live mode; `0` sends the whole population |
| `MAX_ENTITIES` | `0` | Safety cap on entities per upload; `0` means no cap |

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/tenants/` | Register a tenant (`name`, `sector`) |
| `GET` | `/api/v1/tenants/{id}` | Fetch a tenant |
| `POST` | `/api/v1/upload/analyze` | Multipart upload (`tenant_id` form field, `files` parts); returns schema, predictions and playbooks |
| `GET` | `/api/v1/analytics/metrics?tenant_id=` | Aggregated KPIs, charts and rows for the tenant's latest run; `404` until a run exists |
| `GET` | `/api/v1/analytics/status` | Mode, model and batch configuration for the dashboard banner |

Tenant and analysis stores are in memory: restarting the server clears them, and a dashboard
opened for a forgotten tenant redirects to onboarding rather than looping.

## Tests

```bash
python -m pytest -q
```

239 tests, none of which touch the network. They cover the feature math against hand-computed
values (velocity, zero-denominator guards, failure vocabularies, temporal anchoring, noise
exclusion), the mock data generator's reproducibility and cohort separation, schema
resolution for all three sectors, the offline gateway's determinism and tier thresholds, the
sector cores, batch chunking and partial-failure survival, Excel ingestion, and the metrics
and routing layer behind the dashboards.

## Known limitations

- Tailwind and Chart.js load from CDNs, so the UI needs internet access even when scoring is
  offline; the Tailwind CDN also prints a production warning in the console by design.
- The intervention drawer's deploy button copies the playbook payload to the clipboard. No
  delivery channel (email, SMS, push) is wired up in this build, and the UI says so.
- Analysis history is not persisted: only the latest run per tenant is kept, in memory.
