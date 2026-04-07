"""
CrewAI-based multi-agent crew for FinSight.

Roles:
- SEC Research Specialist  – retrieves and summarises SEC filings
- Financial Analyst        – interprets financial data and trends
- Investment Report Writer – produces the final cited research report
"""

from __future__ import annotations

import logging
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI

from config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_AGENT_ITERATIONS,
    OPENAI_API_KEY,
)
from rag.retriever import get_retriever

logger = logging.getLogger(__name__)


# ── RAG tool ──────────────────────────────────────────────────────────────────

class SECRetrieverTool(BaseTool):
    """LangChain-backed RAG tool that searches the SEC filings vector store."""

    name: str = "SEC Filings Retriever"
    description: str = (
        "Search and retrieve relevant excerpts from SEC filings (10-K, 10-Q, 8-K). "
        "Input should be a concise search query about a company or financial topic."
    )

    def _run(self, query: str) -> str:  # noqa: D401
        retriever = get_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant SEC filing excerpts found."
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            citation = (
                f"[{i}] {meta.get('ticker', 'N/A')} "
                f"{meta.get('form', '')} "
                f"({meta.get('filingDate', 'unknown')})"
            )
            parts.append(f"{citation}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


# ── LLM factory ───────────────────────────────────────────────────────────────

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )


# ── Agents ────────────────────────────────────────────────────────────────────

def create_researcher_agent() -> Agent:
    """SEC Research Specialist – retrieves filing excerpts."""
    return Agent(
        role="SEC Research Specialist",
        goal=(
            "Retrieve the most relevant excerpts from SEC filings for the given query. "
            "Always include citation metadata (ticker, form type, filing date)."
        ),
        backstory=(
            "You are a former SEC examiner with 15 years of experience reading 10-K, "
            "10-Q, and 8-K filings. You excel at extracting the most financially "
            "significant information quickly and accurately."
        ),
        tools=[SECRetrieverTool()],
        llm=_llm(),
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )


def create_analyst_agent() -> Agent:
    """Financial Analyst – interprets the retrieved data."""
    return Agent(
        role="Financial Analyst",
        goal=(
            "Analyse the retrieved SEC filing excerpts to identify key financial metrics, "
            "trends, and risks. Provide quantitative and qualitative insights with citations."
        ),
        backstory=(
            "You are a CFA charterholder with expertise in equity research. "
            "You can interpret complex financial statements and regulatory disclosures, "
            "turning raw data into actionable investment intelligence."
        ),
        tools=[SECRetrieverTool()],
        llm=_llm(),
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )


def create_writer_agent() -> Agent:
    """Investment Report Writer – produces the final cited research report."""
    return Agent(
        role="Investment Report Writer",
        goal=(
            "Synthesise the financial analysis into a concise, professional research "
            "report with an executive summary, key findings, risk assessment, and "
            "conclusion. All factual claims must reference citation numbers."
        ),
        backstory=(
            "You are a senior investment research writer for a top-tier asset management "
            "firm. Your reports are known for being clear, accurate, and well-structured, "
            "always backed by primary-source citations."
        ),
        tools=[],
        llm=_llm(),
        max_iter=MAX_AGENT_ITERATIONS,
        verbose=True,
    )


# ── Tasks ─────────────────────────────────────────────────────────────────────

def create_retrieval_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Search for the most relevant SEC filing excerpts related to: '{query}'. "
            "Return the top excerpts with citation metadata (ticker, form, date)."
        ),
        expected_output=(
            "A list of relevant SEC filing excerpts, each labelled with a citation "
            "reference [N] including ticker, form type, and filing date."
        ),
        agent=agent,
    )


def create_analysis_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Using the retrieved SEC filing excerpts, analyse the financial situation "
            f"related to: '{query}'. "
            "Cover: revenue/earnings trends, key financial ratios, risk factors, "
            "and management guidance. Reference all data with citation numbers."
        ),
        expected_output=(
            "A structured financial analysis (3-5 paragraphs) with inline citations [N] "
            "covering: revenue, profitability, risks, and forward-looking statements."
        ),
        agent=agent,
    )


def create_report_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Based on the financial analysis for '{query}', write a professional "
            "investment research report. Include: executive summary, key findings, "
            "risk assessment, and investment conclusion. Preserve all citation references."
        ),
        expected_output=(
            "A polished 400-600 word investment research report in Markdown format, "
            "with clear sections and inline citation references [N]."
        ),
        agent=agent,
    )


# ── Crew ─────────────────────────────────────────────────────────────────────

def build_research_crew(query: str) -> Crew:
    """Assemble the FinSight CrewAI research crew for a given *query*."""
    researcher = create_researcher_agent()
    analyst = create_analyst_agent()
    writer = create_writer_agent()

    tasks = [
        create_retrieval_task(researcher, query),
        create_analysis_task(analyst, query),
        create_report_task(writer, query),
    ]

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def run_crew_research(query: str) -> dict:
    """
    Execute the full CrewAI research pipeline for *query*.

    Returns a dict with: query, final_report (the crew's output string).
    """
    crew = build_research_crew(query)
    result = crew.kickoff()
    return {
        "query": query,
        "final_report": str(result),
    }
