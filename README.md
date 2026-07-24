# Google ADK Stock Market Agent

An AI Agent created using [google/adk-python](https://github.com/google/adk-python) to find and rank the top 10 most active trading stocks today based on total dollar trading volume (**Volume × Current Price**).

---

## 🌟 Key ADK Python Architectural Components

Following the patterns in [google/adk-python](https://github.com/google/adk-python):

1. **`google.adk.Agent`**: Defines root agent (`TopActiveStocksAgent`) with instructions, tools, description, and model configuration.
2. **`google.adk.apps.App`**: Wraps the root agent into an ADK Application (`stock_app`).
3. **`google.adk.Runner`**: Orchestrates event-driven execution using `run_async(...)`.
4. **`google.adk.sessions.InMemorySessionService`**: Manages user state and message histories.
5. **ADK Tools**: Decorator-free Python functions with typed signatures and Google-style docstrings automatically declared as ADK tools:
   - `get_stock_market_data()`
   - `calculate_active_trading_stocks(top_n=10)`
   - `get_top_10_active_stocks()`
   - `get_stock_details(ticker)`

---

## 📊 Methodology & Calculations

$$\text{Dollar Volume} = \text{Trading Volume (shares)} \times \text{Current Price (\$)}$$

The agent calculates $\text{Dollar Volume}$ for all stocks, sorts them in descending order, and extracts the top 10.

---

## 🚀 Usage

### Running the Agent

Default query:

```bash
python3 stock_agent.py
```

Custom query:

```bash
python3 stock_agent.py --query "Find top active stocks"
```

Interactive mode:

```bash
python3 stock_agent.py --interactive
```

---

## 🧪 Testing

Run unit & integration tests:

```bash
python3 -m unittest test_stock_agent.py
```

---

## 📁 File Structure

- [stock_agent.py](file:///Users/haonan/Documents/hn-test-repo/stock_agent.py): Full agent implementation referencing `google/adk-python`.
- [test_stock_agent.py](file:///Users/haonan/Documents/hn-test-repo/test_stock_agent.py): Unittest suite for tools, calculations, ADK App, and Runner.
- [README.md](file:///Users/haonan/Documents/hn-test-repo/README.md): Documentation.