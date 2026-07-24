"""
Stock Market Analysis Agent using Google ADK (Agent Development Kit).

This module defines an ADK agent that retrieves stock trading data (mock/fake data for testing),
calculates dollar volume (volume * current_price), sorts stocks by dollar volume, and returns
the top 10 most active trading stocks.
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any, Optional

from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Fake Stock Market Data Generator
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
    """Format dollar amount into human-readable representation."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.2f} Billion"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.2f} Million"
    return f"${amount:,.2f}"


# ---------------------------------------------------------------------------
# 2. ADK Agent Tools
# ---------------------------------------------------------------------------

def get_stock_market_data() -> List[Dict[str, Any]]:
    """Retrieves raw stock trading data including ticker, company name, volume, and current price.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing stock market data.
    """
    return DEFAULT_FAKE_STOCKS


def calculate_active_trading_stocks(top_n: int = 10) -> List[Dict[str, Any]]:
    """Calculates active trading stocks sorted by total dollar volume (volume * current_price).

    Args:
        top_n (int): Number of top active stocks to return (default is 10).

    Returns:
        List[Dict[str, Any]]: Top N active stocks with rank, volume, price, and calculated dollar volume.
    """
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

    # Sort in descending order by volume * current_price
    sorted_stocks = sorted(processed_stocks, key=lambda x: x["volume_times_price"], reverse=True)

    # Assign rankings
    top_stocks = sorted_stocks[:top_n]
    for idx, item in enumerate(top_stocks, start=1):
        item["rank"] = idx

    return top_stocks


def get_top_10_active_stocks() -> Dict[str, Any]:
    """Finds and returns the top 10 most active trading stocks sorted by volume times current price today.

    Returns:
        Dict[str, Any]: Summary dictionary containing ranking, metrics, and stock entries.
    """
    top_stocks = calculate_active_trading_stocks(top_n=10)
    total_dollar_volume_top_10 = sum(s["volume_times_price"] for s in top_stocks)

    return {
        "title": "Top 10 Most Active Trading Stocks Today (Volume * Current Price)",
        "metric": "Trading Dollar Volume (Volume * Current Price)",
        "total_top_10_dollar_volume": format_dollar_amount(total_dollar_volume_top_10),
        "top_stocks": top_stocks
    }


# ---------------------------------------------------------------------------
# 3. Fallback / Mock LLM Implementation for Offline or Keyless Environments
# ---------------------------------------------------------------------------

class ADKStockLlm(BaseLlm):
    """Custom ADK BaseLlm subclass that executes tool workflow deterministically

    when running without a live Gemini API key connection.
    """
    model: str = "adk-stock-llm"

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
        # Inspect incoming request contents for function response from ADK runner
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
            # First turn: Request tool call to get top 10 active stocks
            call_content = types.Content(
                role="model",
                parts=[types.Part.from_function_call(name="get_top_10_active_stocks", args={})]
            )
            yield LlmResponse(content=call_content, partial=False)
        else:
            # Second turn: Format and return final answer based on tool output
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
# 4. Create and Configure Agent
# ---------------------------------------------------------------------------

def create_stock_agent(model_name: str = "gemini-2.5-flash") -> Agent:
    """Factory function to build and configure the Stock ADK Agent.

    Args:
        model_name (str): Model name or standard ADK model identifier.

    Returns:
        Agent: An initialized google-adk Agent instance.
    """
    has_api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    if has_api_key:
        llm_backend = model_name
    else:
        llm_backend = ADKStockLlm()

    instruction_prompt = (
        "You are a professional financial stock market analysis agent created with Google ADK.\n"
        "Your task is to identify the top 10 most active trading stocks today sorted by total dollar volume "
        "(Volume times Current Price).\n"
        "Use the provided tools `get_top_10_active_stocks` or `calculate_active_trading_stocks` to retrieve data.\n"
        "Present the final results as a clean, beautifully formatted markdown table including Rank, Ticker, "
        "Company Name, Volume, Price, and Total Dollar Volume."
    )

    agent = Agent(
        name="TopActiveStocksAgent",
        model=llm_backend,
        description="ADK Agent that finds the top 10 most active stocks today sorted by volume * current price.",
        instruction=instruction_prompt,
        tools=[get_stock_market_data, calculate_active_trading_stocks, get_top_10_active_stocks],
        sub_agents=[]
    )

    return agent


# ---------------------------------------------------------------------------
# 5. Agent Execution Runner
# ---------------------------------------------------------------------------

async def run_stock_agent_async(query: str, app_name: str = "stock_app", user_id: str = "user_1", session_id: str = "session_1") -> str:
    """Runs the Stock ADK Agent asynchronously with the given query.

    Args:
        query (str): The prompt/query for the agent.
        app_name (str): ADK application identifier.
        user_id (str): User identifier for session.
        session_id (str): Session identifier.

    Returns:
        str: Final text output from the agent.
    """
    agent = create_stock_agent()
    session_service = InMemorySessionService()

    # Ensure session exists
    session_service.create_session_sync(app_name=app_name, user_id=user_id, session_id=session_id)

    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)

    user_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)]
    )

    final_text_parts = []

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_message):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text_parts.append(part.text)

    return "\n".join(final_text_parts)


def run_stock_agent(query: str) -> str:
    """Synchronous wrapper for running the Stock ADK Agent.

    Args:
        query (str): User prompt for the stock agent.

    Returns:
        str: Agent response text.
    """
    return asyncio.run(run_stock_agent_async(query))


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Google ADK Stock Market Agent - Find Top 10 Active Stocks by Volume * Price"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Find the top 10 most active trading stocks today, sorted by volume times current price.",
        help="Query for the stock analysis agent."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive CLI mode."
    )

    args = parser.parse_args()

    if args.interactive:
        print("=== Google ADK Stock Market Agent (Interactive Mode) ===")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_input = input("You > ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue
                print("\n[Agent is processing...]")
                output = run_stock_agent(user_input)
                print(f"\nAgent >\n{output}\n")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print(f"Executing query: '{args.query}'\n")
        result = run_stock_agent(args.query)
        print(result)


if __name__ == "__main__":
    main()
