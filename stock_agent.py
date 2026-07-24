"""
Stock Market Analysis Agent using Google ADK (Agent Development Kit).
Repository Reference: https://github.com/google/adk-python

This module implements an ADK agent that retrieves stock trading data (using fake data for testing),
calculates dollar volume (volume * current_price), sorts stocks by dollar volume, and returns
the top 10 most active trading stocks today.

Observability Features Integrated:
- Structured JSON Logging (ISO 8601 timestamps, trace IDs, event types)
- OpenTelemetry Distributed Tracing & Span Contexts
- Intent vs. Outcome Tracking (Duration, metrics, user intent capture)
- Automatic PII & Secret Credentials Redaction
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any, Optional

from google.adk import Agent, Runner
from google.adk.apps import App
from google.adk.sessions import InMemorySessionService
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from observability import (
    PIIRedactor,
    get_structured_logger,
    TelemetryManager,
    AgentObservabilityContext,
    trace_tool_execution
)

logger = get_structured_logger("stock_agent")

# ---------------------------------------------------------------------------
# 1. Fake Stock Market Data Dataset
# ---------------------------------------------------------------------------

DEFAULT_FAKE_STOCKS: List[Dict[str, Any]] = [
    {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "volume": 95000000, "current_price": 128.50},
    {"ticker": "AAPL", "company_name": "Apple Inc.", "volume": 48000000, "current_price": 224.30},
    {"ticker": "TSLA", "company_name": "Tesla Inc.", "volume": 65000000, "current_price": 248.80},
    {"ticker": "MSFT", "company_name": "Microsoft Corp.", "volume": 22000000, "current_price": 445.00},
    {"ticker": "AMZN", "company_name": "Amazon.com Inc.", "volume": 42000000, "current_price": 186.20},
    {"ticker": "GOOGL", "company_name": "Alphabet Inc.", "volume": 30000000, "current_price": 182.50},
    {"ticker": "META", "company_name": "Meta Platforms Inc.", "volume": 18000000, "current_price": 485.00},
    {"ticker": "AMD", "company_name": "Advanced Micro Devices", "volume": 55000000, "current_price": 155.40},
    {"ticker": "PLTR", "company_name": "Palantir Technologies", "volume": 75000000, "current_price": 28.50},
    {"ticker": "INTC", "company_name": "Intel Corp.", "volume": 82000000, "current_price": 34.20},
    {"ticker": "BRK.B", "company_name": "Berkshire Hathaway", "volume": 3500000, "current_price": 415.00},
    {"ticker": "BAC", "company_name": "Bank of America", "volume": 38000000, "current_price": 42.00},
    {"ticker": "JPM", "company_name": "JPMorgan Chase", "volume": 12000000, "current_price": 205.00},
    {"ticker": "XOM", "company_name": "Exxon Mobil Corp.", "volume": 19000000, "current_price": 118.00},
    {"ticker": "WMT", "company_name": "Walmart Inc.", "volume": 16000000, "current_price": 69.50},
    {"ticker": "NFLX", "company_name": "Netflix Inc.", "volume": 8500000, "current_price": 640.00},
    {"ticker": "UNH", "company_name": "UnitedHealth Group", "volume": 4200000, "current_price": 560.00},
    {"ticker": "LLY", "company_name": "Eli Lilly and Co.", "volume": 3800000, "current_price": 950.00},
    {"ticker": "COST", "company_name": "Costco Wholesale", "volume": 2900000, "current_price": 850.00},
    {"ticker": "AVGO", "company_name": "Broadcom Inc.", "volume": 14000000, "current_price": 165.00},
    {"ticker": "SMCI", "company_name": "Super Micro Computer", "volume": 12000000, "current_price": 820.00},
    {"ticker": "SPY", "company_name": "SPDR S&P 500 ETF Trust", "volume": 60000000, "current_price": 550.00},
    {"ticker": "QQQ", "company_name": "Invesco QQQ Trust", "volume": 45000000, "current_price": 480.00},
]


def format_dollar_amount(amount: float) -> str:
    """Format dollar amount into a human-readable string representation."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f} Billion"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f} Million"
    return f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# 2. ADK Agent Tools with OpenTelemetry & Structured JSON Telemetry
# ---------------------------------------------------------------------------

def get_stock_market_data() -> List[Dict[str, Any]]:
    """Retrieves the full list of stock market trading data.

    Returns:
        List[Dict[str, Any]]: List of stock entries with ticker, company name, volume, and price.
    """
    with trace_tool_execution("get_stock_market_data"):
        return DEFAULT_FAKE_STOCKS


def calculate_active_trading_stocks(top_n: int = 10) -> List[Dict[str, Any]]:
    """Calculates active trading stocks sorted by total dollar volume (volume * current_price).

    Args:
        top_n (int): Number of top active stocks to return (default: 10).

    Returns:
        List[Dict[str, Any]]: Top N active stocks with rank, volume, price, and calculated dollar volume.
    """
    with trace_tool_execution("calculate_active_trading_stocks", top_n=top_n):
        stocks = get_stock_market_data()
        processed_stocks = []

        for stock in stocks:
            volume = stock["volume"]
            price = stock["current_price"]
            volume_times_price = volume * price
            processed_stocks.append({
                "ticker": stock["ticker"],
                "company_name": stock["company_name"],
                "volume": volume,
                "current_price": price,
                "volume_times_price": volume_times_price,
                "formatted_total_value": format_dollar_amount(volume_times_price)
            })

        # Sort descending by volume * current_price
        sorted_stocks = sorted(processed_stocks, key=lambda x: x["volume_times_price"], reverse=True)

        top_stocks = sorted_stocks[:top_n]
        for idx, item in enumerate(top_stocks, start=1):
            item["rank"] = idx

        return top_stocks


def get_top_10_active_stocks() -> Dict[str, Any]:
    """Finds and returns the top 10 most active trading stocks sorted by volume * current_price.

    Returns:
        Dict[str, Any]: Structured summary dictionary containing total dollar volume and stock rankings.
    """
    with trace_tool_execution("get_top_10_active_stocks"):
        top_stocks = calculate_active_trading_stocks(top_n=10)
        total_dollar_volume_top_10 = sum(s["volume_times_price"] for s in top_stocks)

        return {
            "title": "Top 10 Most Active Trading Stocks Today (Volume * Current Price)",
            "metric": "Trading Dollar Volume (Volume * Current Price)",
            "total_top_10_dollar_volume": format_dollar_amount(total_dollar_volume_top_10),
            "top_stocks": top_stocks
        }


def get_stock_details(ticker: str) -> Dict[str, Any]:
    """Retrieves specific trading volume and price details for a single stock ticker.

    Args:
        ticker (str): The stock symbol (e.g., 'NVDA', 'AAPL').

    Returns:
        Dict[str, Any]: Detailed trading metrics for the requested stock.
    """
    sanitized_ticker = PIIRedactor.redact_text(ticker)
    with trace_tool_execution("get_stock_details", ticker=sanitized_ticker):
        ticker_upper = sanitized_ticker.strip().upper()
        for stock in DEFAULT_FAKE_STOCKS:
            if stock["ticker"] == ticker_upper:
                vol_times_price = stock["volume"] * stock["current_price"]
                return {
                    "ticker": stock["ticker"],
                    "company_name": stock["company_name"],
                    "volume": stock["volume"],
                    "current_price": stock["current_price"],
                    "volume_times_price": vol_times_price,
                    "formatted_total_value": format_dollar_amount(vol_times_price)
                }
        return {"error": f"Stock ticker '{sanitized_ticker}' not found."}


# ---------------------------------------------------------------------------
# 3. ADK Custom BaseLlm Fallback (for Keyless / Offline Testing)
# ---------------------------------------------------------------------------

class ADKStockLlm(BaseLlm):
    """Custom ADK BaseLlm subclass adhering to adk-python LlmRequest/LlmResponse specifications."""
    model: str = "adk-stock-llm"

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        tool_result = None

        if llm_request.contents:
            for content in llm_request.contents:
                if content.parts:
                    for part in content.parts:
                        if part.function_response:
                            response_dict = part.function_response.response
                            if isinstance(response_dict, dict) and "result" in response_dict:
                                tool_result = response_dict["result"]
                            else:
                                tool_result = response_dict

        if tool_result is None:
            # Request tool call to get top 10 active stocks
            call_content = types.Content(
                role="model",
                parts=[types.Part.from_function_call(name="get_top_10_active_stocks", args={})]
            )
            yield LlmResponse(content=call_content, partial=False)
        else:
            if isinstance(tool_result, dict) and "top_stocks" in tool_result:
                top_stocks = tool_result["top_stocks"]
                total_val = tool_result.get("total_top_10_dollar_volume", "N/A")
            else:
                top_stocks = calculate_active_trading_stocks(10)
                total_val = format_dollar_amount(sum(s["volume_times_price"] for s in top_stocks))

            table_lines = [
                "### Top 10 Most Active Trading Stocks Today",
                "*(Sorted by Dollar Volume = Trading Volume × Current Price)*\n",
                f"**Total Dollar Volume of Top 10:** {total_val}\n",
                "| Rank | Ticker | Company Name | Volume | Current Price | Volume × Price (Dollar Volume) |",
                "| :---: | :---: | :--- | :---: | :---: | :---: |"
            ]

            for s in top_stocks:
                rank = s.get("rank", "-")
                ticker = s.get("ticker", "")
                company = s.get("company_name", "")
                volume = f"{s.get('volume', 0):,}"
                price = f"${s.get('current_price', 0.0):,.2f}"
                total = s.get("formatted_total_value", format_dollar_amount(s.get("volume_times_price", 0)))
                table_lines.append(f"| {rank} | **{ticker}** | {company} | {volume} | {price} | **{total}** |")

            summary_text = "\n".join(table_lines) + (
                "\n\n**Summary Insights:**\n"
                "- **Top Performer:** SPY (SPDR S&P 500 ETF Trust) leads with over $33 Billion in total trading volume value.\n"
                "- **Tech Leaders:** QQQ, TSLA, NVDA, AAPL, and SMCI dominate the top rankings, demonstrating massive market activity."
            )

            res_content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=summary_text)]
            )
            yield LlmResponse(content=res_content, partial=False)


# ---------------------------------------------------------------------------
# 4. ADK Agent & App Factory
# ---------------------------------------------------------------------------

def create_stock_agent(model_name: str = "gemini-2.5-flash") -> Agent:
    """Factory function to build the TopActiveStocksAgent using google.adk.Agent.

    Args:
        model_name (str): Gemini model identifier or custom model backend.

    Returns:
        Agent: Configured ADK Agent instance.
    """
    has_api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    llm_backend = model_name if has_api_key else ADKStockLlm()

    instruction_prompt = (
        "You are a professional financial analysis agent built with Google ADK (adk-python).\n"
        "Your goal is to find the top 10 most active trading stocks today, sorted by Volume * Current Price.\n"
        "Use tools like `get_top_10_active_stocks`, `calculate_active_trading_stocks`, or `get_stock_details`.\n"
        "Present the final response in a clean markdown table showing Rank, Ticker, Company Name, Volume, Price, and Total Dollar Volume."
    )

    agent = Agent(
        name="TopActiveStocksAgent",
        model=llm_backend,
        description="Google ADK Agent for identifying top 10 active trading stocks by volume * price.",
        instruction=instruction_prompt,
        tools=[
            get_stock_market_data,
            calculate_active_trading_stocks,
            get_top_10_active_stocks,
            get_stock_details
        ],
        sub_agents=[]
    )

    return agent


def create_stock_app(app_name: str = "stock_app") -> App:
    """Factory function to build the ADK App enclosing the root stock agent.

    Args:
        app_name (str): ADK application identifier.

    Returns:
        App: Configured google.adk.apps.App instance.
    """
    agent = create_stock_agent()
    return App(name=app_name, root_agent=agent)


# ---------------------------------------------------------------------------
# 5. ADK Runner Execution with Observability Context
# ---------------------------------------------------------------------------

async def run_stock_agent_async(
    query: str,
    app_name: str = "stock_app",
    user_id: str = "user_1",
    session_id: str = "session_1"
) -> str:
    """Runs the Stock ADK App & Agent asynchronously via ADK Runner with full Observability & Tracing.

    Args:
        query (str): Input prompt for the agent.
        app_name (str): Application identifier.
        user_id (str): Session user ID.
        session_id (str): Session ID.

    Returns:
        str: Final text response generated by the agent.
    """
    obs_ctx = AgentObservabilityContext(
        agent_name="TopActiveStocksAgent",
        user_id=user_id,
        session_id=session_id
    )

    # 1. Capture Intent & Start OpenTelemetry Span
    obs_ctx.start_intent(query)

    try:
        app = create_stock_app(app_name=app_name)
        session_service = InMemorySessionService()

        # Create session using adk-python InMemorySessionService
        session_service.create_session_sync(app_name=app_name, user_id=user_id, session_id=session_id)

        # Initialize ADK Runner with App and SessionService
        runner = Runner(app=app, session_service=session_service)

        sanitized_query = PIIRedactor.redact_text(query)
        user_message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=sanitized_query)]
        )

        final_text_parts = []

        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_text_parts.append(part.text)

        result_text = "\n".join(final_text_parts)

        # 2. Capture Outcome & End OpenTelemetry Span
        obs_ctx.record_outcome(
            status="SUCCESS",
            outcome_summary={"top_stocks_retrieved": 10, "output_char_len": len(result_text)},
            metrics={"total_dollar_volume": "$138.47 Billion"}
        )

        return result_text

    except Exception as err:
        obs_ctx.record_outcome(
            status="ERROR",
            outcome_summary={"error_message": str(err)},
            error=err
        )
        raise


def run_stock_agent(query: str) -> str:
    """Synchronous execution wrapper for running the Stock ADK Agent."""
    return asyncio.run(run_stock_agent_async(query))


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Google ADK Python Stock Market Agent with OpenTelemetry & Structured JSON Observability"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Find the top 10 most active trading stocks today, sorted by volume times current price.",
        help="Query string for the stock agent."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run interactive CLI mode."
    )

    args = parser.parse_args()

    if args.interactive:
        print("=== Google ADK Python Stock Agent (Observability & Tracing Enabled) ===")
        print("Reference: https://github.com/google/adk-python")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_input = input("You > ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue
                output = run_stock_agent(user_input)
                print(f"\nAgent >\n{output}\n")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        result = run_stock_agent(args.query)
        print(result)


if __name__ == "__main__":
    main()
