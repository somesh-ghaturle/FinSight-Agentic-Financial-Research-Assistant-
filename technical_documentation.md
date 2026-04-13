# FinSight Technical Documentation

## 1. Project Overview

FinSight is an agentic financial research assistant for SEC filings. It combines two orchestration styles, LangGraph and CrewAI, with a retrieval-augmented generation pipeline backed by either FAISS or Pinecone. The system ingests SEC EDGAR filings, chunks and indexes them, retrieves relevant excerpts for user questions, and produces cited research reports. It also logs quality metrics to MLflow so outputs can be evaluated over time.

The application is split into a FastAPI backend, a Streamlit frontend, reusable RAG utilities, two agent implementations, evaluation helpers, and AWS deployment artifacts.

## 2. High-Level Architecture

The runtime flow is:

1. A user enters a question in the Streamlit UI or calls the FastAPI API directly.
2. The backend resolves the request to either the LangGraph pipeline or the CrewAI crew.
3. The selected agent backend retrieves relevant SEC filing chunks through the shared retriever layer.
4. The model synthesises a financial answer with inline citations referencing retrieved chunks.
5. If evaluation is enabled, the response is scored and logged to MLflow.

The architecture supports two storage backends:

1. FAISS for local persistence.
2. Pinecone for hosted vector storage.

## 3. Repository Layout

```text
agents/
  crewai_agents.py        CrewAI research crew, tasks, and RAG tool
  langgraph_agents.py     LangGraph retrieval -> analysis -> writing graph

api/
  main.py                 FastAPI app entry point
  routes.py               Health, ingest, research, and evaluation endpoints
  models.py               Pydantic request and response models

config/
  settings.py             Environment-driven configuration

rag/
  embeddings.py           OpenAI embedding factory
  faiss_store.py          Local FAISS index build/load helpers
  pinecone_store.py       Pinecone upsert/store helpers
  retriever.py            Vector store retriever factory
  sec_fetcher.py          SEC EDGAR fetch logic

evaluation/
  metrics.py              Citation accuracy, hallucination rate, faithfulness
  mlflow_tracker.py       MLflow logging and experiment summary

ui/
  app.py                  Streamlit interface

aws/
  cloudformation.yaml     ECS/Fargate deployment stack
  ecs_task_definition.json ECS task definition example

tests/
  test_api.py             API tests
  test_agents.py          Agent pipeline tests
  test_metrics.py         Metric tests
  test_rag.py             RAG tests
```

## 4. Runtime Components

### 4.1 FastAPI backend

The backend starts in [api/main.py](api/main.py) and exposes the application under `/api/v1`.

Key behavior:

1. CORS is enabled for all origins, methods, and headers.
2. Swagger UI is available at `/docs` and ReDoc at `/redoc`.
3. The root route returns a simple service summary with links to docs and health.

### 4.2 Streamlit frontend

The UI in [ui/app.py](ui/app.py) is a conversational front end. It:

1. Checks backend health.
2. Lets the user ingest SEC filings for a ticker.
3. Lets the user choose the agent backend (`langgraph` or `crewai`).
4. Displays research reports, citations, and evaluation metrics.
5. Reads the API base URL from `STREAMLIT_API_URL`.

### 4.3 Shared RAG layer

The RAG subsystem is centered on [rag/sec_fetcher.py](rag/sec_fetcher.py), [rag/retriever.py](rag/retriever.py), [rag/faiss_store.py](rag/faiss_store.py), [rag/pinecone_store.py](rag/pinecone_store.py), and [rag/embeddings.py](rag/embeddings.py).

The ingestion path is:

1. Resolve ticker to CIK.
2. Fetch recent filing metadata from SEC EDGAR.
3. Download the filing document text.
4. Split the text into chunks.
5. Embed the chunks.
6. Store them in FAISS or Pinecone with citation metadata.

### 4.4 Agent backends

There are two interchangeable research backends:

1. LangGraph in [agents/langgraph_agents.py](agents/langgraph_agents.py).
2. CrewAI in [agents/crewai_agents.py](agents/crewai_agents.py).

Both backends use the same retriever and OpenAI chat model configuration.

### 4.5 Evaluation and observability

Evaluation is implemented in [evaluation/metrics.py](evaluation/metrics.py) and [evaluation/mlflow_tracker.py](evaluation/mlflow_tracker.py). The project tracks:

1. Citation accuracy.
2. Hallucination rate.
3. Faithfulness.

These metrics are logged to MLflow when the research endpoint is called with evaluation enabled.

## 5. FastAPI API Contract

### 5.1 Routes

All API routes are mounted under `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Return system status and configuration summary |
| POST | `/research` | Run the research agent pipeline |
| POST | `/ingest` | Fetch and index SEC filings |
| GET | `/evaluation/summary` | Return aggregate MLflow metrics |

### 5.2 Request Models

Defined in [api/models.py](api/models.py).

`QueryRequest`:

1. `query` is required and must be at least 3 characters.
2. `backend` defaults to `langgraph` and accepts `langgraph` or `crewai`.
3. `evaluate` defaults to `true`.

`IngestRequest`:

1. `ticker` is required.
2. `form_types` defaults to `10-K`.
3. `max_filings` defaults to `5` and is constrained to the range 1 to 20.

### 5.3 Response Models

`QueryResponse` includes:

1. The original query.
2. The final report.
3. Optional analysis text.
4. A citation list with metadata.
5. Optional evaluation results.

`IngestResponse` includes:

1. The ticker.
2. Number of filings ingested.
3. Approximate chunks indexed.
4. A status message.

`HealthResponse` includes:

1. `status`.
2. `vector_store_backend`.
3. `llm_model`.

`EvaluationSummaryResponse` includes:

1. Total run count.
2. Average citation accuracy.
3. Average hallucination rate.
4. Average faithfulness.

## 6. Agent Execution Details

### 6.1 LangGraph pipeline

The LangGraph workflow in [agents/langgraph_agents.py](agents/langgraph_agents.py) is a three-node graph:

1. Retriever node.
2. Analyst node.
3. Writer node.

Flow:

1. The retriever node fetches relevant documents and extracts citation metadata.
2. The analyst node turns filing excerpts into structured financial analysis.
3. The writer node rewrites the analysis into a polished investment report.

The graph state stores the query, retrieved documents, analysis, final report, citations, and messages.

### 6.2 CrewAI pipeline

The CrewAI implementation in [agents/crewai_agents.py](agents/crewai_agents.py) uses:

1. A SEC Research Specialist agent.
2. A Financial Analyst agent.
3. An Investment Report Writer agent.

The crew executes sequentially and uses a shared RAG tool backed by the same retriever abstraction.

### 6.3 Shared LLM and embeddings

Both agent backends use:

1. `LLM_MODEL` for the chat model.
2. `LLM_TEMPERATURE` for generation randomness.
3. `OPENAI_API_KEY` for authentication.
4. `EMBEDDING_MODEL` for vector embeddings.

## 7. RAG and Storage

### 7.1 SEC data acquisition

[rag/sec_fetcher.py](rag/sec_fetcher.py) is the SEC EDGAR integration layer. It:

1. Resolves ticker symbols to CIKs using SEC company ticker data.
2. Retrieves recent filing metadata for supported forms.
3. Downloads filing text from SEC archives.
4. Retries on 429 responses with exponential backoff.

Supported filing forms are:

1. `10-K`
2. `10-Q`
3. `8-K`

### 7.2 Chunking and citation metadata

Chunking is handled by the FAISS and Pinecone modules with the same splitter settings:

1. `CHUNK_SIZE`
2. `CHUNK_OVERLAP`

Every chunk stores metadata such as ticker, form, filing date, accession number, and chunk index. This metadata is what later powers citation rendering in the UI and report output.

### 7.3 FAISS backend

[rag/faiss_store.py](rag/faiss_store.py) supports:

1. Building an index from filing dictionaries.
2. Loading an existing persisted index from disk.
3. Building on demand if no index exists yet.

The index persists under `data/faiss_index`.

### 7.4 Pinecone backend

[rag/pinecone_store.py](rag/pinecone_store.py) supports:

1. Creating the Pinecone index if it does not exist.
2. Upserting chunked documents with metadata.
3. Returning a vector store connected to the existing index.

The Pinecone index name, dimension, and environment are controlled via settings.

### 7.5 Retriever abstraction

[rag/retriever.py](rag/retriever.py) chooses FAISS or Pinecone based on `VECTOR_STORE_BACKEND`, then returns a LangChain retriever with `RETRIEVER_TOP_K` search depth.

## 8. Evaluation and MLflow

### 8.1 Metrics

[evaluation/metrics.py](evaluation/metrics.py) defines three scoring functions:

1. `citation_accuracy(report, citations)` checks whether citation IDs in the report exist in the returned citation list.
2. `hallucination_rate(report, context_docs)` estimates how many report sentences are not grounded in the retrieved context.
3. `faithfulness(report, context_docs)` is defined as `1 - hallucination_rate`.

The helper `evaluate_response(...)` returns a consolidated metric dictionary.

### 8.2 MLflow tracking

[evaluation/mlflow_tracker.py](evaluation/mlflow_tracker.py) configures the MLflow tracking URI, ensures the experiment exists, and logs:

1. Query parameters.
2. Number of retrieved docs.
3. Extra params such as backend name.
4. Citation accuracy.
5. Hallucination rate.
6. Faithfulness.
7. Citation count.
8. Report length.

The `/evaluation/summary` endpoint returns averages across the experiment.

## 9. Configuration

All settings are loaded in [config/settings.py](config/settings.py). The environment file template is [/.env.example](.env.example).

### 9.1 LLM and embeddings

1. `OPENAI_API_KEY` is required for LLM and embedding calls.
2. `LLM_MODEL` defaults to `gpt-4o-mini`.
3. `LLM_TEMPERATURE` defaults to `0.0`.
4. `EMBEDDING_MODEL` defaults to `text-embedding-3-small`.

### 9.2 Vector store

1. `VECTOR_STORE_BACKEND` defaults to `faiss`.
2. `PINECONE_API_KEY` is only needed when Pinecone is selected.
3. `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME`, and `PINECONE_DIMENSION` control hosted vector-store behavior.

### 9.3 SEC EDGAR

1. `SEC_EDGAR_USER_AGENT` is sent with SEC requests.
2. `SEC_MAX_FILINGS` limits how many filings are fetched per ingest request.
3. `CHUNK_SIZE` and `CHUNK_OVERLAP` control document splitting.

### 9.4 API, UI, and MLflow

1. `API_HOST`, `API_PORT`, and `API_RELOAD` configure the FastAPI server.
2. `STREAMLIT_API_URL` configures the UI backend URL.
3. `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` configure metrics logging.

### 9.5 Agent tuning

1. `MAX_AGENT_ITERATIONS` caps CrewAI iteration depth.
2. `RETRIEVER_TOP_K` controls retrieval breadth.

## 10. Local Development

### 10.1 Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 10.2 Configure environment

```bash
cp .env.example .env
```

At minimum, set `OPENAI_API_KEY`. Set `VECTOR_STORE_BACKEND=pinecone` only if Pinecone credentials are available.

### 10.3 Run the API

```bash
uvicorn api.main:app --reload
```

### 10.4 Run the UI

```bash
streamlit run ui/app.py
```

### 10.5 Run tests

```bash
pytest tests/ -v
```

## 11. Docker and Compose

The Dockerfile uses a two-stage build:

1. A builder stage installs Python dependencies.
2. A runtime stage copies the installed packages, copies the application code, creates the FAISS data directory, and switches to a non-root user.

The compose file runs three services:

1. `api` on port 8000.
2. `ui` on port 8501.
3. `mlflow` on port 5000.

The compose setup wires the UI to the API and mounts volumes for FAISS and MLflow persistence.

```bash
docker compose up --build
```

## 12. AWS Deployment

The AWS deployment artifacts support ECS Fargate with an Application Load Balancer.

### 12.1 CloudFormation stack

[aws/cloudformation.yaml](aws/cloudformation.yaml) provisions:

1. ECS roles and policies.
2. Security groups.
3. CloudWatch log groups.
4. ECS cluster and task definition.
5. Application Load Balancer and target groups.
6. ECS service with API and UI containers.

### 12.2 ECS task definition example

[aws/ecs_task_definition.json](aws/ecs_task_definition.json) shows a task layout with:

1. API container.
2. UI container.
3. Optional MLflow container.
4. Secrets injected from AWS Secrets Manager.
5. CloudWatch logging configuration.

### 12.3 Deployment inputs

Typical deployment inputs include:

1. VPC and subnet IDs.
2. ECR image URIs for the API and UI.
3. Secrets Manager ARNs for OpenAI and Pinecone keys.
4. Vector store backend choice.
5. LLM model name.

## 13. Testing Strategy

The tests are organized by subsystem:

1. API route tests in [tests/test_api.py](tests/test_api.py).
2. Agent pipeline tests in [tests/test_agents.py](tests/test_agents.py).
3. RAG tests in [tests/test_rag.py](tests/test_rag.py).
4. Metric tests in [tests/test_metrics.py](tests/test_metrics.py).

The project is structured so the core dependencies can be mocked for deterministic tests.

## 14. Operational Notes

1. SEC requests depend on a valid user agent string and may be rate-limited.
2. FAISS is local and simpler for development; Pinecone is better for shared or remote deployments.
3. Evaluation logging is optional and only happens when the research request sets `evaluate` to true.
4. The project expects OpenAI credentials for both generation and embeddings.
5. The report quality is heavily dependent on the freshness and coverage of the indexed filings.

## 15. Known Behavioral Characteristics

1. The ingest endpoint approximates chunk counts from total text length and `CHUNK_SIZE`.
2. The evaluation heuristics are lightweight and rule-based rather than model-based.
3. The LangGraph pipeline returns both analysis and final report text, while the CrewAI backend returns the crew output as the final report.
4. The UI persists chat history in Streamlit session state.

## 16. Suggested Usage Patterns

1. Use FAISS for local development and quick iteration.
2. Use Pinecone for production-like shared persistence.
3. Use LangGraph if you want explicit node-by-node control over the workflow.
4. Use CrewAI if you prefer role-based agent orchestration.
5. Enable evaluation when you want to track quality regressions across queries.

## 17. File-Specific Reference Map

1. [api/main.py](api/main.py) defines the app and server startup.
2. [api/routes.py](api/routes.py) defines the REST surface.
3. [api/models.py](api/models.py) defines request and response schemas.
4. [agents/langgraph_agents.py](agents/langgraph_agents.py) defines the graph-based pipeline.
5. [agents/crewai_agents.py](agents/crewai_agents.py) defines the CrewAI pipeline.
6. [rag/sec_fetcher.py](rag/sec_fetcher.py) fetches SEC filings.
7. [rag/faiss_store.py](rag/faiss_store.py) manages local vector storage.
8. [rag/pinecone_store.py](rag/pinecone_store.py) manages hosted vector storage.
9. [rag/retriever.py](rag/retriever.py) chooses the retriever implementation.
10. [evaluation/metrics.py](evaluation/metrics.py) calculates response quality.
11. [evaluation/mlflow_tracker.py](evaluation/mlflow_tracker.py) logs and summarizes metrics.
12. [ui/app.py](ui/app.py) renders the Streamlit app.
13. [config/settings.py](config/settings.py) centralizes environment variables.
14. [docker-compose.yml](docker-compose.yml) runs the local multi-service stack.
15. [Dockerfile](Dockerfile) packages the application for container use.
16. [aws/cloudformation.yaml](aws/cloudformation.yaml) deploys the AWS infrastructure.
17. [aws/ecs_task_definition.json](aws/ecs_task_definition.json) documents the ECS task layout.

## 18. Summary

FinSight is a modular SEC research system with a clean separation between ingestion, retrieval, generation, evaluation, UI, and deployment. The codebase is designed so the agent backend can be swapped without changing the API contract, and the storage backend can be switched between local FAISS and cloud Pinecone through configuration alone.