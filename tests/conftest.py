"""
conftest.py – stubs for heavy optional dependencies so that tests can run
in a minimal environment without crewai, langgraph, or pinecone installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── crewai ────────────────────────────────────────────────────────────────────
if "crewai" not in sys.modules:
    crewai_mod = _stub("crewai")
    crewai_mod.Agent = MagicMock
    crewai_mod.Task = MagicMock
    crewai_mod.Crew = MagicMock
    crewai_mod.Process = MagicMock(sequential="sequential")
    sys.modules["crewai"] = crewai_mod
    sys.modules["crewai.tools"] = _stub("crewai.tools", BaseTool=object)

# ── langgraph ─────────────────────────────────────────────────────────────────
if "langgraph" not in sys.modules:
    langgraph_mod = _stub("langgraph")

    class _MockStateGraph:
        def __init__(self, state_type=None):
            self._compiled = MagicMock()
            self._compiled.invoke.return_value = {
                "query": "test",
                "final_report": "report",
                "analysis": "analysis",
                "citations": [],
                "messages": [],
            }

        def add_node(self, *args, **kwargs):
            return self

        def add_edge(self, *args, **kwargs):
            return self

        def compile(self):
            return self._compiled

    langgraph_graph = _stub(
        "langgraph.graph",
        StateGraph=_MockStateGraph,
        END="__end__",
        START="__start__",
    )
    sys.modules["langgraph"] = langgraph_mod
    sys.modules["langgraph.graph"] = langgraph_graph

# ── langchain_pinecone ────────────────────────────────────────────────────────
if "langchain_pinecone" not in sys.modules:
    sys.modules["langchain_pinecone"] = _stub("langchain_pinecone", PineconeVectorStore=MagicMock)

# ── pinecone ──────────────────────────────────────────────────────────────────
if "pinecone" not in sys.modules:
    sys.modules["pinecone"] = _stub("pinecone", Pinecone=MagicMock, ServerlessSpec=MagicMock)
