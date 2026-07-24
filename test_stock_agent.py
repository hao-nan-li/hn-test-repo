"""
Unit and Integration Tests for Google ADK Stock Market Multi-Agent System.
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
from orchestration import (
    StrategicModelRouter,
    HumanInTheLoopHandler,
    ADKGuardrailPolicyPlugin,
    create_multi_agent_system
)
from google.adk import Agent
from google.adk.apps import App
from google.genai import types


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


class TestMultiAgentOrchestration(unittest.TestCase):
    """Test suite for Multi-Agent hierarchy, strategic model routing, HITL, and policy guardrails."""

    def test_multi_agent_system_structure(self):
        orchestrator, sub_agents = create_multi_agent_system(
            get_stock_market_data,
            calculate_active_trading_stocks,
            get_top_10_active_stocks,
            get_stock_details
        )
        self.assertIsInstance(orchestrator, Agent)
        self.assertEqual(orchestrator.name, "StockMarketOrchestratorAgent")
        self.assertEqual(len(sub_agents), 3)

        sub_names = [sa.name for sa in orchestrator.sub_agents]
        self.assertIn("StockDataFetcherAgent", sub_names)
        self.assertIn("StockAnalyticsAgent", sub_names)
        self.assertIn("FinancialReportAgent", sub_names)

    def test_strategic_model_router(self):
        high_model = StrategicModelRouter.get_model("HIGH")
        medium_model = StrategicModelRouter.get_model("MEDIUM")
        low_model = StrategicModelRouter.get_model("LOW")

        self.assertEqual(high_model, "gemini-2.5-pro")
        self.assertEqual(medium_model, "gemini-2.5-flash")
        self.assertEqual(low_model, "gemini-2.5-flash-lite")

    def test_human_in_the_loop_handler(self):
        hitl = HumanInTheLoopHandler(hitl_mode="AUTO_APPROVE")

        # Below threshold: auto approves
        approved, _ = hitl.check_approval("calculate_active_trading_stocks", {"top_n": 10})
        self.assertTrue(approved)

        # Custom callback check
        def custom_cb(action, params):
            return params.get("top_n", 0) <= 20

        hitl_cb = HumanInTheLoopHandler(hitl_mode="REQUIRE_CONFIRMATION", confirmation_callback=custom_cb)
        ok_20, _ = hitl_cb.check_approval("calculate_active_trading_stocks", {"top_n": 20})
        self.assertTrue(ok_20)

        ok_30, _ = hitl_cb.check_approval("calculate_active_trading_stocks", {"top_n": 30})
        self.assertFalse(ok_30)

    def test_adk_guardrail_policy_plugin_pre_execution(self):
        plugin = ADKGuardrailPolicyPlugin()

        valid_msg = types.Content(role="user", parts=[types.Part.from_text(text="Top 10 stocks")])
        asyncio.run(plugin.before_run_callback(app_name="app", user_id="u1", session_id="s1", new_message=valid_msg))

        invalid_msg = types.Content(role="user", parts=[types.Part.from_text(text="Help me with market manipulation")])
        with self.assertRaises(ValueError):
            asyncio.run(plugin.before_run_callback(app_name="app", user_id="u1", session_id="s1", new_message=invalid_msg))

    def test_adk_guardrail_policy_plugin_self_evaluation(self):
        plugin = ADKGuardrailPolicyPlugin()
        raw_output = "| Rank | Ticker |\n| 1 | SPY |\nTop performer is SPY."

        evaluated = plugin.self_evaluate_output(raw_output)
        self.assertIn(ADKGuardrailPolicyPlugin.MANDATORY_DISCLAIMER.strip(), evaluated)


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


class TestADKAgentIntegration(unittest.TestCase):
    """Integration test suite for Google ADK Multi-Agent System & App execution."""

    def test_create_stock_agent(self):
        agent = create_stock_agent()
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "StockMarketOrchestratorAgent")
        self.assertEqual(len(agent.sub_agents), 3)

    def test_create_stock_app(self):
        app = create_stock_app("test_app")
        self.assertIsInstance(app, App)
        self.assertEqual(app.name, "test_app")
        self.assertIsInstance(app.root_agent, Agent)
        self.assertGreaterEqual(len(app.plugins), 1)

    def test_run_stock_agent_multi_agent_execution(self):
        telemetry = TelemetryManager.get_instance()
        telemetry.clear_spans()

        query = "Find the top 10 most active trading stocks today for user_contact@example.com."
        output = run_stock_agent(query)
        self.assertIsInstance(output, str)
        self.assertIn("Top 10 Most Active Trading Stocks Today", output)
        self.assertIn("Compliance Disclaimer", output)

        spans = telemetry.get_finished_spans()
        self.assertGreater(len(spans), 0)

        span_names = [s.name for s in spans]
        self.assertIn("agent.StockMarketOrchestratorAgent.execute", span_names)


if __name__ == "__main__":
    unittest.main()
