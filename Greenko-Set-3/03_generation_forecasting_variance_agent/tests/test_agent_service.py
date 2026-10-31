import pytest
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.model_service import ModelService
from backend.variance_engine import VarianceEngine
from backend.agent_service import AgentService, KnowledgeRetriever, DOMAIN_KNOWLEDGE_BASE


@pytest.fixture(scope="module")
def agent():
    ms = ModelService()
    ve = VarianceEngine(ms.df, ms.train_dates)
    return AgentService(ms, ve)


class TestAgentService:
    def test_tfidf_retriever(self):
        retriever = KnowledgeRetriever(DOMAIN_KNOWLEDGE_BASE)
        results = retriever.retrieve("temperature derating module efficiency", top_k=2)
        assert len(results) > 0
        top_topic = results[0]['document']['topic']
        assert "Temperature" in top_topic or "Derating" in top_topic
        assert results[0]['score'] > 0

    def test_hybrid_retriever_handles_common_typos(self):
        retriever = KnowledgeRetriever(DOMAIN_KNOWLEDGE_BASE)
        # Character n-grams should preserve useful matches when a user mistypes
        # domain terms; this guards the fuzzy-retrieval improvement.
        results = retriever.retrieve("temprature derating silicon efficency", top_k=1)
        assert results
        assert results[0]["document"]["id"] == "KB-SOP-01"

    def test_retrieval_benchmark_precision_recall(self):
        retriever = KnowledgeRetriever(DOMAIN_KNOWLEDGE_BASE)
        bench = retriever.evaluate_retrieval_benchmark()
        assert bench['test_cases'] == 12
        assert bench['domain_queries'] == 10
        assert bench['recall_at_2'] >= 0.8
        assert bench['mrr'] >= 0.8
        assert bench['precision_at_1'] >= 0.7
        assert bench['out_of_domain_rejection_rate'] == 1.0

    def test_date_parsing_formats(self, agent):
        assert agent.parse_date("variance on 2026-08-15") == "2026-08-15"
        assert agent.parse_date("what happened on August 15?") == "2026-08-15"
        assert agent.parse_date("explain Aug 22") == "2026-08-22"
        assert agent.parse_date("on 5th") == "2026-08-05"
        assert agent.parse_date("random general query") == ""

    def test_date_parsing_beyond_august_2026(self, agent):
        assert agent.parse_date("what about 2027-01-10?") == "2027-01-10"
        assert agent.parse_date("schedule on 2025-12-31") == "2025-12-31"

    def test_query_specific_date(self, agent):
        res = agent.query("Why was generation lower on 2026-08-15?")
        assert res['target_date'] == "2026-08-15"
        assert len(res['reasoning_steps']) >= 4
        assert len(res['retrieval_evidence']) > 0
        assert len(res['recommendations']) > 0
        assert 'drivers' in res
        assert 'waterfall' in res
        assert 'Outage' in res['category']

    def test_query_nonexistent_date(self, agent):
        res = agent.query("Explain variance on 2026-08-99")
        assert res['category'] == "Date Not Found"
        assert "not found" in res['summary'].lower()

    def test_query_outage_audit(self, agent):
        res = agent.query("Summarize all outage events across the portfolio")
        assert res['category'] == "Operational Outage Audit"
        assert len(res['reasoning_steps']) >= 3
        assert len(res['recommendations']) > 0
        assert "Portfolio Outage Audit" in res['summary']

    def test_query_top_misses(self, agent):
        res = agent.query("What were the top 5 forecast misses?")
        assert res['category'] == "Variance Ranking"
        assert len(res['reasoning_steps']) >= 3
        assert "Top 5 Forecast Misses" in res['summary']

    def test_query_model_performance(self, agent):
        res = agent.query("Show machine learning model evaluation and benchmarks")
        assert res['category'] == "Model Performance Audit"
        assert "Operational Day-Ahead Baseline" in res['summary']
        assert "Random Forest Regressor" in res['summary']

    def test_query_regulatory(self, agent):
        res = agent.query("Explain CERC DSM regulations and day-ahead scheduling penalties")
        assert res['category'] == "Regulatory Context"
        assert "Deviation Settlement Mechanism" in res['summary']

    def test_query_general_overview(self, agent):
        res = agent.query("Give me a summary overview of plant generation")
        assert res['category'] == "Portfolio Overview"
        assert "Total Actual Generation" in res['summary']

    def test_query_irrelevant_out_of_domain(self, agent):
        res = agent.query("What is the recipe for chocolate cake?")
        assert res['category'] == "Out of Domain"
        assert "Out of Domain" in res['summary']
        assert len(res['recommendations']) > 0

    def test_fallback_on_empty_query(self, agent):
        res = agent.query("   ")
        assert res['category'] == "Fallback"
        assert res['confidence'] == "low"


def test_optional_llm_uses_deterministic_fallback_without_key(agent, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = agent.query("Why was generation lower on 2026-08-15?")
    assert result["explanation_mode"] == "deterministic_fallback"
    assert result["llm_status"] == "disabled_no_api_key"
    assert result["waterfall"]


def test_optional_llm_rewrites_only_grounded_summary(agent, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": "Grounded explanation with supplied evidence."}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("backend.agent_service.httpx.post", lambda *args, **kwargs: FakeResponse())
    result = agent.query("Why was generation lower on 2026-08-15?")
    assert result["summary"] == "Grounded explanation with supplied evidence."
    assert result["explanation_mode"] == "llm_grounded"
    assert result["llm_status"] == "success"
    # Numerical tool output is preserved separately from LLM narrative synthesis.
    assert result["waterfall"]
    assert result["retrieval_evidence"]
