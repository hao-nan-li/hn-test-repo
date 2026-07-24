"""
Unit and Integration Tests for Google ADK Stock Market Agent.
"""

import unittest
import asyncio
from stock_agent import (
    get_stock_market_data,
    calculate_active_trading_stocks,
    get_top_10_active_stocks,
    format_dollar_amount,
    create_stock_agent,
    run_stock_agent,
    run_stock_agent_async
)
from google.adk import Agent


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
        self.assertGreater(sample["volume"], 0)
        self.assertGreater(sample["current_price"], 0)

    def test_calculate_active_trading_stocks(self):
        top_5 = calculate_active_trading_stocks(top_n=5)
        self.assertEqual(len(top_5), 5)

        # Check sorting order: each item's dollar volume must be >= next item
        for i in range(len(top_5) - 1):
            self.assertGreaterEqual(
                top_5[i]["volume_times_price"],
                top_5[i + 1]["volume_times_price"],
                f"Stock at rank {i+1} ({top_5[i]['ticker']}) has smaller dollar volume than rank {i+2} ({top_5[i+1]['ticker']})"
            )

        # Check math accuracy for first stock
        first = top_5[0]
        expected_val = first["volume"] * first["current_price"]
        self.assertEqual(first["volume_times_price"], expected_val)

    def test_get_top_10_active_stocks(self):
        res = get_top_10_active_stocks()
        self.assertIsInstance(res, dict)
        self.assertIn("top_stocks", res)

        top_10 = res["top_stocks"]
        self.assertEqual(len(top_10), 10)

        # Verify rank 1 ticker is SPY with highest trading dollar volume
        self.assertEqual(top_10[0]["ticker"], "SPY")
        self.assertEqual(top_10[0]["rank"], 1)

    def test_format_dollar_amount(self):
        self.assertEqual(format_dollar_amount(33_000_000_000), "$33.00 Billion")
        self.assertEqual(format_dollar_amount(500_000_000), "$500.00 Million")
        self.assertEqual(format_dollar_amount(1234.56), "$1,234.56")


class TestADKAgentIntegration(unittest.TestCase):
    """Integration test suite for Google ADK Agent initialization and execution."""

    def test_create_stock_agent(self):
        agent = create_stock_agent()
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "TopActiveStocksAgent")
        self.assertGreaterEqual(len(agent.tools), 3)

    def test_run_stock_agent(self):
        query = "Find the top 10 most active trading stocks today, sorted by volume times current price."
        output = run_stock_agent(query)
        self.assertIsInstance(output, str)
        self.assertIn("Top 10 Most Active Trading Stocks Today", output)

        # Check that top tickers appear in output markdown table
        expected_tickers = ["SPY", "QQQ", "TSLA", "NVDA", "AAPL", "SMCI", "MSFT", "META", "AMD", "AMZN"]
        for ticker in expected_tickers:
            self.assertIn(ticker, output, f"Expected ticker '{ticker}' in agent response output.")


if __name__ == "__main__":
    unittest.main()
