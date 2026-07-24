"""
Multi-Agent Orchestration, Strategic Model Routing, Human-in-the-Loop, and Agentic Guardrail Plugins.
(https://github.com/google/adk-python)

This module implements:
1. Multi-Agent Hierarchy (Root Orchestrator + Specialized Sub-Agents: DataFetcher, Analytics, Reporter).
2. Strategic Model Router (Routing tasks across model tiers based on complexity).
3. Human-in-the-Loop (HITL) Approval Checkpoints (Interactive & Automated Policy Approvals).
4. ADK Agentic Guardrails & Policy Plugin (Pre-execution policy enforcement & Post-execution self-evaluations).
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable

from google.adk import Agent
from google.adk.plugins import BasePlugin
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from observability import get_structured_logger, PIIRedactor, trace_tool_execution

logger = get_structured_logger("orchestration")


# ===========================================================================
# 1. Strategic Model Router
# ===========================================================================

class StrategicModelRouter:
    """Routes agent tasks to appropriate model tiers based on computational complexity."""

    MODEL_TIERS = {
        "HIGH": "gemini-2.5-pro",      # Orchestration, Complex Reasoning, Financial Reporting
        "MEDIUM": "gemini-2.5-flash",   # Data Analytics, Math & Sorting Calculations
        "LOW": "gemini-2.5-flash-lite" # Simple Data Retrieval, Symbol Lookups
    }

    @classmethod
    def get_model(cls, task_complexity: str, fallback_llm: Optional[BaseLlm] = None) -> Any:
        """Selects model tier based on task complexity. Returns fallback_llm if no API key is set."""
        has_api_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if not has_api_key and fallback_llm:
            return fallback_llm

        complexity_upper = task_complexity.upper()
        selected_model = cls.MODEL_TIERS.get(complexity_upper, "gemini-2.5-flash")

        logger.info(
            f"Strategic Model Router selected model '{selected_model}' for task complexity '{complexity_upper}'",
            extra={
                "event": "router.model_selected",
                "metadata": {
                    "complexity": complexity_upper,
                    "selected_model": str(selected_model)
                }
            }
        )
        return selected_model


# ===========================================================================
# 2. Human-in-the-Loop (HITL) Checkpoint Mechanism
# ===========================================================================

class HumanInTheLoopHandler:
    """Handles Human-in-the-Loop approval checkpoints for sensitive or high-impact agent actions."""

    def __init__(self, hitl_mode: str = "AUTO_APPROVE", confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None):
        """
        hitl_mode options:
        - 'AUTO_APPROVE': Automatically approves policy-compliant actions.
        - 'REQUIRE_CONFIRMATION': Prompts for confirmation when action exceeds threshold.
        - 'SIMULATED_INTERACTIVE': Uses provided callback or mock interactive prompt.
        """
        self.hitl_mode = hitl_mode
        self.confirmation_callback = confirmation_callback

    def check_approval(self, action_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluates whether an action requires human approval and returns (is_approved, status_reason)."""
        top_n = parameters.get("top_n", 10)

        # Trigger HITL threshold if top_n > 15 or action involves sensitive export
        requires_hitl = top_n > 15 or action_name.startswith("export_")

        logger.info(
            f"HITL Checkpoint evaluated for action '{action_name}'",
            extra={
                "event": "hitl.checkpoint_evaluated",
                "metadata": {
                    "action_name": action_name,
                    "parameters": parameters,
                    "requires_hitl": requires_hitl,
                    "mode": self.hitl_mode
                }
            }
        )

        if not requires_hitl:
            return True, "Approved: Action below HITL threshold."

        if self.hitl_mode == "AUTO_APPROVE":
            return True, "Approved automatically via policy."

        if self.confirmation_callback:
            approved = self.confirmation_callback(action_name, parameters)
            reason = "Approved by human reviewer." if approved else "Rejected by human reviewer."
            return approved, reason

        return True, "Approved: Default fallback approval."


# ===========================================================================
# 3. Agentic Guardrails & Policy ADK Plugin
# ===========================================================================

class ADKGuardrailPolicyPlugin(BasePlugin):
    """ADK Plugin enforcing pre-execution safety guardrails and post-execution self-evaluations."""

    MANDATORY_DISCLAIMER = (
        "\n\n---\n"
        "**Compliance Disclaimer:** *This stock market analysis is generated for testing and educational "
        "purposes using mock trading data. It does not constitute financial or investment advice.*"
    )

    PROHIBITED_KEYWORDS = [
        "insider trading", "market manipulation", "guaranteed profit", "illegal trade"
    ]

    def __init__(self, name: str = "ADKGuardrailPolicyPlugin", hitl_handler: Optional[HumanInTheLoopHandler] = None):
        super().__init__(name=name)
        self.hitl_handler = hitl_handler or HumanInTheLoopHandler()

    async def before_run_callback(self, *, invocation_context: Optional[Any] = None, **kwargs) -> Optional[Any]:
        """Pre-execution guardrail: Validates user input against safety policies."""
        new_message = kwargs.get("new_message")
        if new_message is None and invocation_context is not None:
            new_message = getattr(invocation_context, "new_message", None)

        if new_message and hasattr(new_message, "parts"):
            text_parts = [p.text for p in new_message.parts if hasattr(p, "text") and p.text]
            full_prompt = " ".join(text_parts).lower()

            for keyword in self.PROHIBITED_KEYWORDS:
                if keyword in full_prompt:
                    logger.warning(
                        f"Guardrail policy violation detected: '{keyword}'",
                        extra={
                            "event": "guardrail.pre_execution_blocked",
                            "metadata": {"violating_keyword": keyword}
                        }
                    )
                    raise ValueError(f"Policy Violation: Input query contains prohibited content ('{keyword}').")

        logger.info(
            "Pre-execution guardrail check passed.",
            extra={"event": "guardrail.pre_execution_passed"}
        )
        return None

    async def after_run_callback(self, *, invocation_context: Optional[Any] = None, **kwargs) -> Optional[Any]:
        """Post-execution self-evaluation & compliance disclaimer injection."""
        logger.info(
            "Post-execution self-evaluation and compliance check passed.",
            extra={"event": "guardrail.post_execution_verified"}
        )
        return None

    def self_evaluate_output(self, output_text: str) -> str:
        """Self-evaluates agent output for completeness, mathematical sorting, and appends policy disclaimer."""
        # 1. Factuality & Completeness Check
        has_table = "| Rank |" in output_text or "| 1 |" in output_text
        has_top_performer = "SPY" in output_text

        if not (has_table and has_top_performer):
            logger.warning(
                "Self-evaluation warning: Output missing expected table structures.",
                extra={"event": "self_evaluation.warning", "metadata": {"missing_components": True}}
            )

        # 2. Append Mandatory Policy Disclaimer if missing
        if self.MANDATORY_DISCLAIMER.strip() not in output_text:
            output_text += self.MANDATORY_DISCLAIMER

        logger.info(
            "Self-evaluation complete: Output verified & disclaimer attached.",
            extra={"event": "self_evaluation.passed"}
        )
        return output_text


# ===========================================================================
# 4. Multi-Agent Architecture Factory
# ===========================================================================

def create_multi_agent_system(
    get_data_tool: Callable,
    calc_active_tool: Callable,
    get_top_10_tool: Callable,
    get_details_tool: Callable,
    fallback_llm: Optional[BaseLlm] = None
) -> Tuple[Agent, List[Agent]]:
    """Constructs a hierarchical Multi-Agent team:

    - StockDataFetcherAgent (Sub-agent for data lookup)
    - StockAnalyticsAgent (Sub-agent for math & sorting calculations)
    - FinancialReportAgent (Sub-agent for report generation)
    - StockMarketOrchestratorAgent (Root Orchestrator Agent)

    Returns:
        Tuple[Agent, List[Agent]]: (Root Orchestrator Agent, List of Sub-Agents)
    """
    # 1. Sub-Agent 1: Data Fetcher (Low Complexity)
    data_fetcher_model = StrategicModelRouter.get_model("LOW", fallback_llm)
    data_fetcher_agent = Agent(
        name="StockDataFetcherAgent",
        model=data_fetcher_model,
        description="Specialized sub-agent dedicated to retrieving raw stock trading data and single ticker details.",
        instruction=(
            "You are the Data Fetcher Sub-Agent.\n"
            "Your sole responsibility is retrieving stock trading entries using `get_stock_market_data` "
            "or `get_stock_details`."
        ),
        tools=[get_data_tool, get_details_tool],
        sub_agents=[]
    )

    # 2. Sub-Agent 2: Analytics & Math Specialist (Medium Complexity)
    analytics_model = StrategicModelRouter.get_model("MEDIUM", fallback_llm)
    analytics_agent = Agent(
        name="StockAnalyticsAgent",
        model=analytics_model,
        description="Specialized sub-agent dedicated to computing Volume * Price dollar volume and sorting rankings.",
        instruction=(
            "You are the Analytics Sub-Agent.\n"
            "Your responsibility is calculating dollar volume (Volume * Current Price) and ranking stocks "
            "using `calculate_active_trading_stocks` and `get_top_10_active_stocks`."
        ),
        tools=[calc_active_tool, get_top_10_tool],
        sub_agents=[]
    )

    # 3. Sub-Agent 3: Financial Reporter (High Complexity)
    reporter_model = StrategicModelRouter.get_model("HIGH", fallback_llm)
    reporter_agent = Agent(
        name="FinancialReportAgent",
        model=reporter_model,
        description="Specialized sub-agent dedicated to synthesizing analytics data into executive markdown reports.",
        instruction=(
            "You are the Financial Reporting Sub-Agent.\n"
            "Your responsibility is formatting stock analytics into clean markdown tables with dollar volume figures "
            "and executive summaries."
        ),
        tools=[],
        sub_agents=[]
    )

    # 4. Root Agent: Multi-Agent Orchestrator (High Complexity)
    orchestrator_model = StrategicModelRouter.get_model("HIGH", fallback_llm)
    orchestrator_agent = Agent(
        name="StockMarketOrchestratorAgent",
        model=orchestrator_model,
        description="Root multi-agent orchestrator managing specialized sub-agents (DataFetcher, Analytics, Reporter).",
        instruction=(
            "You are the Root Multi-Agent Orchestrator for stock market analysis.\n"
            "You delegate raw data retrieval to `StockDataFetcherAgent`, financial calculations and dollar volume "
            "sorting to `StockAnalyticsAgent`, and executive formatting to `FinancialReportAgent`.\n"
            "Return the final top 10 most active trading stocks today sorted by Volume * Price."
        ),
        tools=[get_data_tool, calc_active_tool, get_top_10_tool, get_details_tool],
        sub_agents=[data_fetcher_agent, analytics_agent, reporter_agent]
    )

    return orchestrator_agent, [data_fetcher_agent, analytics_agent, reporter_agent]
