# Google ADK Stock Market Agent

An AI Agent created with **Google ADK (Agent Development Kit)** in Python to find and rank the top 10 most active trading stocks today based on total dollar trading volume (**Volume × Current Price**).

---

## 🌟 Key Features

1. **Google ADK Integration**: Built using `google-adk` (`google.adk.Agent`, `google.adk.Runner`, `google.adk.sessions.InMemorySessionService`).
2. **Tool-Driven Analysis**: Equipped with specialized Python tools to fetch stock market data, perform exact volume × price calculations, and sort stock rankings.
3. **Structured Outputs**: Formats stock market activity into clean markdown tables with dollar volume figures (e.g. `$33.00 Billion`).
4. **Flexible LLM Execution**: Supports both standard Gemini models (`gemini-2.5-flash`) when an API key is available, as well as an offline fallback model (`ADKStockLlm`) for keyless/offline testing environments.

---

## 📊 Methodology & Calculations

Stock market trading activity is calculated using:

$$\text{Dollar Volume} = \text{Trading Volume (shares)} \times \text{Current Price (\$)}$$

The agent sorts all stocks in descending order by $\text{Dollar Volume}$ and extracts the top 10 entries.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- `google-adk` package installed (`pip install google-adk`)

### Running the Agent

Run the default query via CLI:

```bash
python3 stock_agent.py
```

Run with a custom query:

```bash
python3 stock_agent.py --query "List the top 10 most active trading stocks today"
```

Run in interactive CLI mode:

```bash
python3 stock_agent.py --interactive
```

---

## 🧪 Running Unit & Integration Tests

Run the full test suite using Python's `unittest`:

```bash
python3 -m unittest test_stock_agent.py
```

---

## 📁 File Structure

- [stock_agent.py](file:///Users/haonan/Documents/hn-test-repo/stock_agent.py): Main agent implementation containing fake data generator, tools, custom LLM fallback, ADK Agent, Runner, and CLI entry point.
- [test_stock_agent.py](file:///Users/haonan/Documents/hn-test-repo/test_stock_agent.py): Comprehensive test suite verifying tool logic, volume * price calculations, ranking order, and ADK agent runner execution.
- [README.md](file:///Users/haonan/Documents/hn-test-repo/README.md): Project overview and usage documentation.