# Google ADK Stock Market Multi-Agent System

An enterprise Multi-Agent System created using [google/adk-python](https://github.com/google/adk-python) to find and rank the top 10 most active trading stocks today based on total dollar trading volume (**Volume × Current Price**).

---

## 🏛️ Multi-Agent Architecture & Orchestration Pattern

The system replaces monolithic agent designs with a **Hierarchical Multi-Agent Architecture**:

```
                  ┌─────────────────────────────────────┐
                  │    StockMarketOrchestratorAgent     │
                  │        (Root ADK Orchestrator)      │
                  └──────────────────┬──────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
│StockDataFetcherAg.│       │StockAnalyticsAgent│       │FinancialReportAg. │
│(Sub-Agent: Data)  │       │(Sub-Agent: Math)  │       │(Sub-Agent: Format)│
└───────────────────┘       └───────────────────┘       └───────────────────┘
```

1. **`StockMarketOrchestratorAgent` (Root Agent)**: Manages task decomposition, delegates work to specialized sub-agents, and synthesizes output.
2. **`StockDataFetcherAgent` (Sub-Agent)**: Dedicated to raw stock market data retrieval (`get_stock_market_data`, `get_stock_details`).
3. **`StockAnalyticsAgent` (Sub-Agent)**: Dedicated to calculating $\text{Volume} \times \text{Current Price}$ dollar volume and ranking top active stocks (`calculate_active_trading_stocks`, `get_top_10_active_stocks`).
4. **`FinancialReportAgent` (Sub-Agent)**: Formats data into executive markdown reports with market insights.

---

## 🎯 Strategic Model Routing

`StrategicModelRouter` dynamically routes sub-agent tasks to appropriate model tiers based on complexity:

- **HIGH Complexity** (`gemini-2.5-pro`): Orchestration & Executive Reporting.
- **MEDIUM Complexity** (`gemini-2.5-flash`): Data Analytics & Dollar Volume Calculations.
- **LOW Complexity** (`gemini-2.5-flash-lite`): Raw Data Lookup & Symbol Fetching.

---

## 🛑 Human-in-the-Loop (HITL) Checkpoints

`HumanInTheLoopHandler` provides approval policy checkpoints for high-impact or large-scale actions:
- Threshold check on parameter limits (e.g. requesting `top_n > 15` or data exports).
- Execution states: `APPROVED`, `REJECTED`, `MODIFIED`.
- Supports automated policy approval and custom interactive callbacks.

---

## 🛡️ Agentic Guardrails & Policy Plugin

`ADKGuardrailPolicyPlugin` subclasses `google.adk.plugins.BasePlugin`:
- **Pre-execution Guardrails**: Scans user queries for prohibited content (`insider trading`, `market manipulation`) before runner execution.
- **Post-execution Self-Evaluation**: Automated check for factuality, sorting math integrity, and mandatory injection of financial compliance disclaimers.

---

## 📊 Observability & Distributed Tracing

- **Structured JSON Logging**: ISO 8601 timestamps, log levels, logger names, event types, trace IDs, and span IDs.
- **OpenTelemetry Tracing**: Distributed context propagation with `TracerProvider` and `InMemorySpanExporter`.
- **Intent vs. Outcome Tracking**: Explicit tracking of raw prompt intent (`agent.intent_captured`), execution duration, output volume, metrics, and completion status (`agent.outcome_captured`).
- **PII & Credentials Redaction (`PIIRedactor`)**: Automatic scrubbing of sensitive emails, phone numbers, SSNs, credit cards, API keys (`AIza...`, `sk-...`), bearer tokens, and IP addresses.

---

## 🚀 Usage

```bash
# Run agent with default query
python3 stock_agent.py

# Run custom query
python3 stock_agent.py --query "Find top active stocks for user@example.com"

# Run interactive mode
python3 stock_agent.py --interactive

# Run unit & integration test suite (17/17 tests passing)
python3 -m unittest test_stock_agent.py
```

---

## 📁 File Structure

- [stock_agent.py](file:///Users/haonan/Documents/hn-test-repo/stock_agent.py): Multi-Agent Root Orchestrator & Runner execution.
- [orchestration.py](file:///Users/haonan/Documents/hn-test-repo/orchestration.py): Multi-agent factory, Strategic Model Router, HITL checkpoints, and ADK Guardrail Plugin.
- [observability.py](file:///Users/haonan/Documents/hn-test-repo/observability.py): Structured JSON logging, OpenTelemetry tracing, intent/outcome tracking, and PII redactor.
- [test_stock_agent.py](file:///Users/haonan/Documents/hn-test-repo/test_stock_agent.py): Unittest suite covering all 17 multi-agent, routing, HITL, policy, and telemetry test cases.
- [README.md](file:///Users/haonan/Documents/hn-test-repo/README.md): System documentation.