# FinSight – Agentic Financial Research Assistant

> End-to-end multi-agent system using **LangGraph** and **CrewAI** that retrieves SEC filings
> via RAG (**FAISS** / **Pinecone**), synthesises cited financial summaries served through
> **FastAPI** + **Streamlit** on **AWS ECS Fargate**; agent observability and evaluation
> pipeline (citation accuracy, hallucination rate, faithfulness) tracked in **MLflow**.

---

## Architecture

```
User (Browser)
      │
      ▼
┌─────────────┐   REST   ┌───────────────────────────────────────────┐
│  Streamlit  │◄────────►│            FastAPI  /api/v1               │
│   (ui/)     │          │  ┌──────────────┐   ┌──────────────────┐  │
└─────────────┘          │  │ LangGraph    │   │  CrewAI Crew     │  │
                         │  │  Pipeline    │   │  (researcher,    │  │
                         │  │ retriever ──►│   │   analyst,       │  │
                         │  │ analyst   ──►│   │   writer)        │  │
                         │  │ writer    ──►│   └──────────────────┘  │
                         │  └──────────────┘                         │
                         │           │                               │
                         │           ▼                               │
                         │  ┌─────────────────┐                      │
                         │  │  RAG Pipeline   │                      │
                         │  │  FAISS/Pinecone │◄── SEC EDGAR API     │
                         │  └─────────────────┘                      │
                         │           │                               │
                         │           ▼                               │
                         │  ┌─────────────────┐                      │
                         │  │  MLflow Tracker │                      │
                         │  │  - citation acc │                      │
                         │  │  - hallucination│                      │
                         │  │  - faithfulness │                      │
                         │  └─────────────────┘                      │
                         └───────────────────────────────────────────┘
```

## Project Structure

```
FinSight/
├── agents/
│   ├── langgraph_agents.py   # LangGraph retriever→analyst→writer graph
│   └── crewai_agents.py      # CrewAI researcher, analyst, writer crew
├── rag/
│   ├── sec_fetcher.py        # SEC EDGAR API client
│   ├── embeddings.py         # OpenAI embedding factory
│   ├── faiss_store.py        # FAISS vector store (local)
│   ├── pinecone_store.py     # Pinecone vector store (cloud)
│   └── retriever.py          # Unified retriever factory
├── api/
│   ├── main.py               # FastAPI app entry point
│   ├── routes.py             # /research, /ingest, /health, /evaluation/summary
│   └── models.py             # Pydantic request/response models
├── ui/
│   └── app.py                # Streamlit chat interface
├── evaluation/
│   ├── metrics.py            # Citation accuracy, hallucination rate, faithfulness
│   └── mlflow_tracker.py     # MLflow logging & experiment summary
├── config/
│   └── settings.py           # All settings via environment variables
├── aws/
│   ├── ecs_task_definition.json  # ECS Fargate task definition
│   └── cloudformation.yaml       # Full AWS CloudFormation stack
├── tests/
│   ├── test_metrics.py       # Evaluation metrics unit tests
│   ├── test_api.py           # FastAPI route tests (mocked)
│   ├── test_agents.py        # LangGraph agent tests (mocked)
│   └── test_rag.py           # RAG pipeline tests (mocked)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env – at minimum set OPENAI_API_KEY
```

### 3. Ingest SEC filings

```bash
# Start the API first
uvicorn api.main:app --reload

# Ingest Apple's latest 10-K via the API
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "form_types": ["10-K"], "max_filings": 3}'
```

### 4. Run a research query

```bash
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are Apple'\''s main revenue risks?", "backend": "langgraph", "evaluate": true}'
```

### 5. Launch the Streamlit UI

```bash
streamlit run ui/app.py
# Open http://localhost:8501
```

### 6. View MLflow metrics

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

## Docker Compose (all services)

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
```

| Service  | URL                       |
|----------|---------------------------|
| API      | http://localhost:8000/docs |
| UI       | http://localhost:8501      |
| MLflow   | http://localhost:5000      |

## AWS Deployment

```bash
# 1. Push images to ECR
aws ecr create-repository --repository-name finsight-api
aws ecr create-repository --repository-name finsight-ui
docker build -t finsight-api . && docker push <ecr-uri>/finsight-api:latest
docker build -t finsight-ui . && docker push <ecr-uri>/finsight-ui:latest

# 2. Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file aws/cloudformation.yaml \
  --stack-name finsight-prod \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    VpcId=vpc-xxxx \
    SubnetIds=subnet-aaa,subnet-bbb \
    ECRApiImage=<ecr-uri>/finsight-api:latest \
    ECRUiImage=<ecr-uri>/finsight-ui:latest \
    OpenAIApiKeySecretArn=arn:aws:secretsmanager:... \
    PineconeApiKeySecretArn=arn:aws:secretsmanager:...
```

## API Reference

| Method | Endpoint                      | Description                               |
|--------|-------------------------------|-------------------------------------------|
| GET    | `/api/v1/health`              | System health & configuration             |
| POST   | `/api/v1/ingest`              | Fetch & index SEC filings for a ticker    |
| POST   | `/api/v1/research`            | Run multi-agent research pipeline         |
| GET    | `/api/v1/evaluation/summary`  | Aggregate MLflow evaluation metrics       |

Full interactive docs at `/docs` (Swagger UI) or `/redoc`.

## Evaluation Metrics

| Metric              | Definition                                                       |
|---------------------|------------------------------------------------------------------|
| `citation_accuracy` | Fraction of cited [N] references that map to retrieved documents |
| `hallucination_rate`| Fraction of report sentences not grounded in retrieved context   |
| `faithfulness`      | 1 − hallucination_rate (RAGAS-style)                             |

All metrics are logged to MLflow after each `/research` call (when `evaluate: true`).

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

All settings are read from environment variables. See `.env.example` for the full list.

| Variable               | Default                    | Description                         |
|------------------------|----------------------------|-------------------------------------|
| `OPENAI_API_KEY`       | –                          | OpenAI API key (required)           |
| `VECTOR_STORE_BACKEND` | `faiss`                    | `faiss` or `pinecone`               |
| `PINECONE_API_KEY`     | –                          | Pinecone API key                    |
| `LLM_MODEL`            | `gpt-4o-mini`              | OpenAI chat model                   |
| `MLFLOW_TRACKING_URI`  | `sqlite:///mlflow.db`      | MLflow backend store URI            |
| `SEC_MAX_FILINGS`      | `5`                        | Max filings to ingest per request   |
| `RETRIEVER_TOP_K`      | `5`                        | Number of chunks to retrieve        |

