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

## Two scoring engines, chosen per run

The dashboard offers both scorers side by side and you pick one at the moment you analyze:

- **Analyze with AI model** — calls the hosted model your key resolved to (`openai/gpt-oss-120b`
  on Groq's free tier, by default).
- **Analyze with system model** — a deterministic local scorer. It satisfies the same contract
  as the live client but computes every score from the uploaded feature values, so it needs no
  credentials and makes no network calls. Scores are a pure function of the input: the same
  upload always yields the same numbers, in seconds.

Running the same files through both is the point — the AI button and the system button sit next
to each other so the two can be compared without restarting the server or editing config. Each
stored run records which engine produced it, and the dashboard banner, the schema panel and the
customer detail page all report that run's engine rather than a global setting.

The system engine is what makes the whole platform — all three dashboards included — usable with
no key at all. Two consequences worth knowing:

- The system scorer normalises signals across the batch it is handed, so it is deliberately run
  in a single pass (`BATCH_SIZE=0`) rather than chunked. Slicing the population would
  re-normalise each slice against its own worst customer.
- With no key configured the AI button is disabled and the note under it says why, rather than
  silently scoring on the other engine.

To enable the AI engine, put one key in `api_key.env` (or `.env`). The host that supplies the key
also picks the endpoint and a model its free plan serves, so one line is enough:

```bash
# Groq — free tier, no credit card, key from the API Keys page at console.groq.com
GROQ_API_KEY=your_groq_key_here
```

`HF_TOKEN` (Hugging Face's Inference Providers router), `OPENROUTER_API_KEY` and
`DASHSCOPE_API_KEY` work the same way; `api_key.env.example` lists all four with their
free-tier limits and default models. Any other OpenAI-compatible host — Google's Gemini free
tier included — works by setting `AI_API_KEY` together with `QWEN_BASE_URL` and `QWEN_MODEL`.

`QWEN_MODE` decides whether the AI engine exists, not which engine a run uses — that is the
button. `live` builds the AI client unconditionally and fails at startup if no key resolves, so a
deployment that was meant to call a model cannot quietly fall back to local scoring; `mock`
switches the AI engine off even when a key is present, leaving only the system model. The default
`auto` enables the AI engine when a key exists and makes it the default choice.

### Measuring the model you configured

The mock datasets have a known churning cohort, so a hosted model's output can be measured
rather than eyeballed:

```bash
python evaluate_ai_provider.py --sector saas --entities 60
```

Per sector it prints coverage, cohort separation, AUC, tier consistency and how many replies
the app had to repair — beside the same numbers for the offline oracle — and gates the run on
coverage and AUC. Free tiers meter tokens, so the prompt size is printed before you spend it.

Measured on Groq's free plan with 30 SaaS customers: `openai/gpt-oss-120b` scored every one of
them in six calls, AUC 1.000, separation 0.542 against the offline oracle's 0.446, and put all
25 of the generated churning cohort in HIGH or CRITICAL with no false alarms — it passes the
gate. The alternatives on the same key do not: `qwen/qwen3.6-27b` returns an empty generation
that fails Groq's JSON validation, `qwen/qwen3.8-27b` needs ~1,900 output tokens for ten
customers against a 1,000-per-minute ceiling that rejects the request outright, and
`openai/gpt-oss-20b` degenerates into stringified array elements after the first prediction.
That is why `gpt-oss-120b` at six customers per call is the configured Groq default.

Some repair is normal. A hosted open-weight model answers `"risk_tier": "critical"` where the
dashboards match `CRITICAL` exactly, quotes numbers the domain model wants as floats, and —
most often — names a tier that contradicts its own probability, since a tier here is defined by
`RISK_THRESHOLDS` and the model was never told the bands. `infrastructure/ai/normalise_prediction.py`
fixes all three at the boundary, so a sloppy reply degrades one score instead of emptying the
at-risk KPI or labelling a CRITICAL customer MEDIUM.

## Project structure

```text
├── churn_platform/
│   ├── application/           # Use cases and DTOs (no framework imports)
│   │   ├── dtos/              #   AnalysisResponse, MetricsSummary
│   │   └── use_cases/         #   schema resolution, synthesis, batching, summarising
│   ├── domain/                # Models and interfaces (no dependencies)
│   ├── infrastructure/
│   │   ├── ai/                #   live gateway, offline gateway, sector cores, prompts,
│   │   │                      #   reply normalisation
│   │   ├── parsers/           #   ingestion, feature synthesis, sentiment, enrichers,
│   │   │                      #   bundled sample data
│   │   └── repositories/      #   in-memory tenant and analysis stores
│   ├── presentation/
│   │   ├── api/v1/            #   tenants, upload/analyze, upload/demo-data, analytics
│   │   ├── static/            #   app.js, styles.css
│   │   └── templates/         #   base, one dashboard per sector, customer detail
│   ├── config.py              # Settings from environment / .env
│   └── main.py                # FastAPI app and sector routing
├── data/                      # Seeded sample datasets (fintech, saas, telecom), committed and
│                              #   bundled so the dashboard's Load sample data button can serve them
├── tests/                     # pytest suite, no network access
├── generate_mock_data.py      # Seeded mock dataset generator
├── evaluate_ai_provider.py    # Scores a configured model against the known cohort
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

Every key in the example is commented out, so copying it leaves the AI engine unavailable and
the system model as the only choice. Uncomment one — `GROQ_API_KEY` is the free option that
needs no credit card — to enable the AI engine.

### 5. Generate mock data (optional)

```bash
python generate_mock_data.py
```

Writes four CSVs per sector into `data/`. The generator is seeded: in every sector the first
25 entities carry deliberate churn patterns (collapsed usage, failed payments, dropped calls
on one tower, draining balances) and the remaining 75 are healthy, which is what makes the
dashboards show real separation instead of noise.

You do not have to run it to try the platform. Those CSVs are committed and bundled with the
deployment, and every dashboard has a **Load sample data** button that fetches the four exports
for the current tenant's sector straight into the upload panel — no file picker, and no local
filesystem needed on a serverless host. It loads the files and stops there, so you still choose
which engine scores them. Run the generator only to regenerate them.

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
| `GROQ_API_KEY` | unset | Live client against Groq's free tier (`api.groq.com/openai/v1`) |
| `HF_TOKEN` | unset | Live client against Hugging Face's Inference Providers router |
| `OPENROUTER_API_KEY` | unset | Live client against OpenRouter's free models |
| `DASHSCOPE_API_KEY` / `ALIBABA_API_KEY` | unset | Live client against Alibaba Cloud Model Studio (Qwen) |
| `AI_API_KEY` | unset | Key for any other OpenAI-compatible host; requires `QWEN_BASE_URL` |
| `QWEN_MODE` | `auto` | Whether the AI engine is built at all: `auto` enables it when a key exists and makes it the default choice, `mock` disables it even with a key, `live` requires one and raises at startup if none resolves. Which engine a run uses is the dashboard button |
| `QWEN_MODEL` | the key's host default | Model name — `openai/gpt-oss-120b` on Groq, `qwen-max` on DashScope |
| `QWEN_BASE_URL` | the key's host | OpenAI-compatible base URL |
| `QWEN_TIMEOUT_SECONDS` | `120` | Per-request timeout; free tiers can be slow to first token |
| `QWEN_MAX_RETRIES` | `4` | Retries with backoff on `429` / `5xx`, which is how a throttled free tier behaves |
| `BATCH_SIZE` | the key's host default, else `10` | Customers per scoring call in live mode; `0` sends the whole population. Groq's free plan caps output at 1,000 tokens a minute and *rejects* a request that would exceed it, so its default is `6` |
| `MAX_ENTITIES` | `0` | Safety cap on entities per upload; `0` means no cap |

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/tenants/` | Register a tenant (`name`, `sector`) |
| `GET` | `/api/v1/tenants/{id}` | Fetch a tenant |
| `POST` | `/api/v1/upload/analyze` | Multipart upload (`tenant_id` form field, `files` parts, optional `engine` field of `ai`, `system` or `auto`); returns schema, predictions and playbooks, plus the `offline_mode` flag of the engine that actually scored them. `400` if the requested engine is unknown or not configured |
| `GET` | `/api/v1/upload/demo-data?tenant_id=` | The bundled sample exports for that tenant's sector as base64, all in one response; `404` for an unknown tenant or when `data/` is empty |
| `GET` | `/api/v1/analytics/metrics?tenant_id=` | Aggregated KPIs, charts and rows for the tenant's latest run; `404` until a run exists |
| `GET` | `/api/v1/analytics/status` | Model and batch configuration, plus `ai_available` and `default_engine` so the dashboard can decide whether the AI button is clickable |
| `GET` | `/api/v1/analytics/customer?tenant_id=&entity_id=` | One customer's score, rank, per-feature percentile evidence and playbook; `404` when the entity is not in the latest run |

Tenant and analysis stores are in memory: restarting the server clears them, and a dashboard
opened for a forgotten tenant redirects to onboarding rather than looping.

## Tests

```bash
python -m pytest -q
```

340 tests, none of which touch the network — the live-gateway tests mount a stub
chat-completions app into the client's own transport, so retries and reply parsing are
exercised offline, and `tests/conftest.py` stops `Settings` reading `api_key.env` before any
test module is imported, so a real key on disk cannot turn the suite into metered live calls.
They cover the feature math against hand-computed values (velocity,
zero-denominator guards, failure vocabularies, temporal anchoring, noise exclusion), the mock
data generator's reproducibility and cohort separation, schema resolution for all three
sectors, the offline gateway's determinism and tier thresholds, the sector cores, batch
chunking and partial-failure survival, Excel ingestion, the metrics and routing layer behind
the dashboards, the per-customer percentile and ranking math, which provider a key selects,
and a throttled, fence-wrapped reply being parsed and repaired into the vocabulary the
dashboards expect.

The engine matrix is tested as a pure function of `Settings`, so every combination of
`QWEN_MODE` and key-presence is asserted without monkeypatching module state — including that
`live` with no key still raises rather than degrading to local scoring, that a run records its
own engine and not the deployment default, and that requesting an unavailable engine is refused
with an actionable message. The sample-data tests go further than checking bytes come back: they
feed the served exports straight through `ingest()` and then back into `POST /analyze`, proving
the bundled CSVs drive the real pipeline rather than just the file listing.

## Known limitations

- Tailwind and Chart.js load from CDNs, so the UI needs internet access even when scoring is
  offline; the Tailwind CDN also prints a production warning in the console by design.
- The intervention drawer's deploy button copies the playbook payload to the clipboard. No
  delivery channel (email, SMS, push) is wired up in this build, and the UI says so.
- Analysis history is not persisted: only the latest run per tenant is kept, in memory.
- Free hosted tiers serve stock public models. Weights you tuned yourself need a paid
  endpoint or a local runtime, so what is tunable here is the prompt, the batch size and the
  model choice — `evaluate_ai_provider.py` is what tells you whether a change helped.
- A free tier is token-metered, so scoring is slow: a real 100-customer SaaS run on Groq's free
  plan took about six minutes (one schema call plus 17 batches of six) where the offline gateway
  needs about two seconds. `POST /api/v1/upload/analyze` blocks for the whole run — there is no
  job queue — so the browser waits with the button disabled. `MAX_ENTITIES` and `BATCH_SIZE` are
  the levers that keep an upload inside the budget.
- The meter is also daily, not just per minute: Groq's free plan allows 200,000 tokens a day per
  model, and once it is spent every further request is refused with a `429` until it resets.
  Retries cannot fix that, so a batch that hits it is dropped and the run continues with the
  rest — which is why the response says how many entities went unscored instead of failing
  quietly.
- A hosted deployment cannot wait six minutes. `vercel.json` sets `maxDuration: 60`, so on Vercel
  the AI engine will time out on a full customer base while the system engine answers in seconds;
  raising the cap needs a paid plan. The note under the engine buttons says this before you click
  rather than letting the request die unexplained.
- The live model is more trigger-happy than the offline oracle: on the same 100-customer base it
  put 51 customers in HIGH or CRITICAL against a generated cohort of 25. It ranked the cohort
  first (AUC 1.000, top four rows all churning) but flags roughly twice as many customers for
  intervention, which is a precision cost the dashboards do not hide.
