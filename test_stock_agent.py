"""
Unit and Integration Tests for Google ADK Stock Market Agent & Observability/Tracing Architecture.
(https://github.com/google/adk-python)
"""

import unittest
import json
import logging
import asyncio
from stock_agent import (
    get_stock_market_data,
    calculate_active_trading_stocks,
    get_top_10_active_stocks,
    get_stock_details,
    format_dollar_amount,
    create_stock_agent,
    create_stock_app,
    run_stock_agent,
    run_stock_agent_async
)
from observability import (
    PIIRedactor,
    JSONFormatter,
    TelemetryManager,
    AgentObservabilityContext,
    trace_tool_execution,
    get_structured_logger
)
from google.adk import Agent
from google.adk.apps import App


class TestStockAgentTools(unittest.TestCase):
    """Test suite for stock agent tools and data calculations."""

    def test_get_stock_market_data(self):
        data = get_stock_market_data()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 10)

        sample = data[0]
        self.assertIn("ticker", sample)
        self.assertIn("company_name", sample)
        self.assertIn("volume", sample)
        self.assertIn("current_price", sample)

    def test_calculate_active_trading_stocks(self):
        top_5 = calculate_active_trading_stocks(top_n=5)
        self.assertEqual(len(top_5), 5)

        for i in range(len(top_5) - 1):
            self.assertGreaterEqual(
                top_5[i]["volume_times_price"],
                top_5[i + 1]["volume_times_price"],
                f"Stock at rank {i+1} has smaller value than rank {i+2}"
            )

        first = top_5[0]
        self.assertEqual(first["volume_times_price"], first["volume"] * first["current_price"])

    def test_get_top_10_active_stocks(self):
        res = get_top_10_active_stocks()
        self.assertIsInstance(res, dict)
        self.assertIn("top_stocks", res)

        top_10 = res["top_stocks"]
        self.assertEqual(len(top_10), 10)
        self.assertEqual(top_10[0]["ticker"], "SPY")
        self.assertEqual(top_10[0]["rank"], 1)

    def test_get_stock_details(self):
        nvda = get_stock_details("NVDA")
        self.assertEqual(nvda["ticker"], "NVDA")
        self.assertEqual(nvda["company_name"], "NVIDIA Corporation")

        missing = get_stock_details("UNKNOWN_TICKER")
        self.assertIn("error", missing)

    def test_format_dollar_amount(self):
        self.assertEqual(format_dollar_amount(33_000_000_000), "$33.00 Billion")
        self.assertEqual(format_dollar_amount(500_000_000), "$500.00 Million")
        self.assertEqual(format_dollar_amount(1234.56), "$1,234.56")


class TestObservabilityAndTracing(unittest.TestCase):
    """Test suite for Observability, OpenTelemetry Tracing, Intent/Outcome Tracking, and PII Redaction."""

    def setUp(self):
        self.telemetry = TelemetryManager.get_instance()
        self.telemetry.clear_spans()

    def test_pii_redaction(self):
        raw_text = "Contact alice@example.com or call 555-123-4567 with ssn 123-45-6789 and key AIzaSyA1234567890123456789012345678901."
        redacted = PIIRedactor.redact_text(raw_text)
        self.assertNotIn("alice@example.com", redacted)
        self.assertNotIn("555-123-4567", redacted)
        self.assertNotIn("123-45-6789", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)
        self.assertIn("[REDACTED_SSN]", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)

    def test_pii_object_redaction(self):
        raw_obj = {
            "user": "john.doe@domain.com",
            "metadata": ["192.168.1.1", "secret: my_password_123"]
        }
        redacted = PIIRedactor.redact_obj(raw_obj)
        self.assertIn("[REDACTED_EMAIL]", redacted["user"])
        self.assertIn("[REDACTED_IP]", redacted["metadata"][0])

    def test_structured_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="User query for test@test.com",
            args=(),
            exc_info=None
        )
        setattr(record, "event", "agent.intent_captured")
        setattr(record, "agent_name", "TestAgent")

        formatted_json = formatter.format(record)
        parsed = json.loads(formatted_json)

        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["event"], "agent.intent_captured")
        self.assertEqual(parsed["agent_name"], "TestAgent")
        self.assertIn("[REDACTED_EMAIL]", parsed["message"])
        self.assertIn("trace_id", parsed)
        self.assertIn("span_id", parsed)
        self.assertIn("timestamp", parsed)

    def test_opentelemetry_tracing_and_intent_outcome_capture(self):
        obs_ctx = AgentObservabilityContext(agent_name="TestAgent", user_id="user_123", session_id="session_456")
        span = obs_ctx.start_intent("Find top stocks for user@example.com")

        with trace_tool_execution("get_top_10_active_stocks", agent_name="TestAgent"):
            get_top_10_active_stocks()

        obs_ctx.record_outcome("SUCCESS", {"count": 10}, {"dollar_volume": "$138.47 Billion"})

        spans = self.telemetry.get_finished_spans()
        self.assertGreaterEqual(len(spans), 2)

        span_names = [s.name for s in spans]
        self.assertIn("agent.TestAgent.execute", span_names)
        self.assertIn("tool.get_top_10_active_stocks", span_names)

        agent_span = next(s for s in spans if s.name == "agent.TestAgent.execute")
        self.assertEqual(agent_span.attributes["agent.name"], "TestAgent")
        self.assertEqual(agent_span.attributes["session.id"], "session_456")
        self.assertEqual(agent_span.attributes["outcome.status"], "SUCCESS")
        self.assertIn("[REDACTED_EMAIL]", agent_span.attributes["intent.query"])


class TestADKAgentIntegration(unittest.TestCase):
    """Integration test suite for Google ADK Agent and App execution with Observability."""

    def test_create_stock_agent(self):
        agent = create_stock_agent()
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "TopActiveStocksAgent")
        self.assertGreaterEqual(len(agent.tools), 4)

    def test_create_stock_app(self):
        app = create_stock_app("test_app")
        self.assertIsInstance(app, App)
        self.assertEqual(app.name, "test_app")
        self.assertIsInstance(app.root_agent, Agent)

    def test_run_stock_agent_with_observability(self):
        telemetry = TelemetryManager.get_instance()
        telemetry.clear_spans()

        query = "Find the top 10 most active trading stocks today for user_contact@example.com."
        output = run_stock_agent(query)
        self.assertIsInstance(output, str)
        self.assertIn("Top 10 Most Active Trading Stocks Today", output)

        spans = telemetry.get_finished_spans()
        self.assertGreater(len(spans), 0)

        # Check OpenTelemetry trace capture for stock agent run
        span_names = [s.name for s in spans]
        self.assertIn("agent.TopActiveStocksAgent.execute", span_names)


if __name__ == "__main__":
    unittest.main()
