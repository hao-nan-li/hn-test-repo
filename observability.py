"""
Observability, Distributed Tracing, Intent/Outcome Tracking, and PII Redaction Module.

This module provides enterprise-grade observability for Google ADK agents:
1. Structured JSON Logging (ISO 8601 timestamps, trace IDs, event types).
2. Distributed Tracing via OpenTelemetry (TracerProvider, Span attributes, InMemorySpanExporter).
3. Intent vs. Outcome Capture (capturing user intent, execution duration, output metrics).
4. Automatic PII & Credentials Redaction (emails, phone numbers, SSNs, credit cards, API keys, IPs).
"""

import os
import sys
import re
import json
import time
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Callable
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# ===========================================================================
# 1. PII & Sensitive Data Redaction Engine
# ===========================================================================

class PIIRedactor:
    """Engine for redacting Personally Identifiable Information (PII) and secret credentials."""

    PATTERNS = [
        # Email Addresses
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
        # US / International Phone Numbers
        (re.compile(r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'), '[REDACTED_PHONE]'),
        # Social Security Numbers (SSN)
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),
        # Credit Card Numbers
        (re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'), '[REDACTED_CREDIT_CARD]'),
        # Google / Generic API Keys (e.g., AIza..., sk-...)
        (re.compile(r'\b(?:AIza[0-9A-Za-z-_]{20,}|sk-[a-zA-Z0-9_-]{20,})\b'), '[REDACTED_API_KEY]'),
        # Bearer Tokens & Secrets in key=value format (requiring explicit = or :)
        (re.compile(r'(?i)\b(bearer|secret|token|api_key|api-key|password)\b\s*[:=]\s*["\']?([^"\'\s]+)["\']?'), r'\1=[REDACTED_SECRET]'),
        # IPv4 Addresses
        (re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'), '[REDACTED_IP]'),
    ]

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redact sensitive patterns in text string."""
        if not isinstance(text, str):
            return text
        for pattern, replacement in cls.PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    @classmethod
    def redact_obj(cls, data: Any) -> Any:
        """Recursively redact sensitive patterns in dictionaries, lists, and strings."""
        if isinstance(data, str):
            return cls.redact_text(data)
        elif isinstance(data, dict):
            return {k: cls.redact_obj(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.redact_obj(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(cls.redact_obj(item) for item in data)
        return data


# ===========================================================================
# 2. OpenTelemetry Tracing Manager
# ===========================================================================

class TelemetryManager:
    """Manager for OpenTelemetry tracer, span processors, and exporter."""

    _instance: Optional['TelemetryManager'] = None

    def __init__(self, service_name: str = "google-adk-stock-agent"):
        self.service_name = service_name
        self.provider = TracerProvider()
        self.memory_exporter = InMemorySpanExporter()
        self.provider.add_span_processor(SimpleSpanProcessor(self.memory_exporter))
        trace.set_tracer_provider(self.provider)
        self.tracer = trace.get_tracer(service_name)

    @classmethod
    def get_instance(cls, service_name: str = "google-adk-stock-agent") -> 'TelemetryManager':
        if cls._instance is None:
            cls._instance = cls(service_name)
        return cls._instance

    def get_tracer(self) -> trace.Tracer:
        return self.tracer

    def get_finished_spans(self) -> List[Any]:
        return self.memory_exporter.get_finished_spans()

    def clear_spans(self):
        self.memory_exporter.clear()


# ===========================================================================
# 3. Structured JSON Formatter & Logger
# ===========================================================================

class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        current_span = trace.get_current_span()
        span_ctx = current_span.get_span_context() if current_span else None

        trace_id = format(span_ctx.trace_id, '032x') if span_ctx and span_ctx.is_valid else getattr(record, 'trace_id', '0'*32)
        span_id = format(span_ctx.span_id, '016x') if span_ctx and span_ctx.is_valid else getattr(record, 'span_id', '0'*16)

        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, 'event', 'log_message'),
            "trace_id": trace_id,
            "span_id": span_id,
            "message": PIIRedactor.redact_text(record.getMessage()),
        }

        # Include structured context fields if present
        for attr in ('agent_name', 'session_id', 'user_id', 'intent', 'outcome', 'duration_ms', 'metadata'):
            if hasattr(record, attr):
                val = getattr(record, attr)
                log_data[attr] = PIIRedactor.redact_obj(val)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_structured_logger(name: str = "adk_stock_agent", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if re-called
    if not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# ===========================================================================
# 4. Intent vs. Outcome & Tool Tracing Context Manager
# ===========================================================================

class AgentObservabilityContext:
    """High-level observability context manager to track intent, outcome, tool spans, and PII sanitization."""

    def __init__(self, agent_name: str, user_id: str = "user_1", session_id: str = "session_1"):
        self.agent_name = agent_name
        self.user_id = user_id
        self.session_id = session_id
        self.logger = get_structured_logger()
        self.telemetry = TelemetryManager.get_instance()
        self.tracer = self.telemetry.get_tracer()
        self.span: Optional[Span] = None
        self.start_time: float = 0.0

    def start_intent(self, query: str) -> Span:
        """Captures user intent, starts OpenTelemetry span, and outputs structured INTENT log."""
        self.start_time = time.time()
        sanitized_query = PIIRedactor.redact_text(query)

        self.span = self.tracer.start_span(f"agent.{self.agent_name}.execute")
        self.span.set_attribute("agent.name", self.agent_name)
        self.span.set_attribute("session.id", self.session_id)
        self.span.set_attribute("user.id", self.user_id)
        self.span.set_attribute("intent.query", sanitized_query)

        self.logger.info(
            f"Captured user intent for agent '{self.agent_name}'",
            extra={
                "event": "agent.intent_captured",
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "intent": {
                    "raw_query": sanitized_query,
                    "intent_type": "top_active_stocks_analysis",
                }
            }
        )
        return self.span

    def record_outcome(self, status: str, outcome_summary: Any, metrics: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        """Captures agent outcome, records OpenTelemetry status/duration, and outputs structured OUTCOME log."""
        duration_ms = round((time.time() - self.start_time) * 1000, 2)
        sanitized_summary = PIIRedactor.redact_obj(outcome_summary)
        sanitized_metrics = PIIRedactor.redact_obj(metrics or {})

        if self.span:
            self.span.set_attribute("outcome.status", status)
            self.span.set_attribute("outcome.duration_ms", duration_ms)
            if error:
                self.span.record_exception(error)
                self.span.set_status(Status(StatusCode.ERROR, str(error)))
            else:
                self.span.set_status(Status(StatusCode.OK))
            self.span.end()

        log_level = logging.ERROR if status == "ERROR" or error else logging.INFO
        self.logger.log(
            log_level,
            f"Captured outcome for agent '{self.agent_name}': status={status}",
            extra={
                "event": "agent.outcome_captured",
                "agent_name": self.agent_name,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "duration_ms": duration_ms,
                "outcome": {
                    "status": status,
                    "summary": sanitized_summary,
                    "metrics": sanitized_metrics,
                    "error": str(error) if error else None
                }
            }
        )


@contextmanager
def trace_tool_execution(tool_name: str, agent_name: str = "TopActiveStocksAgent", **kwargs):
    """Context manager for tracing tool executions with OpenTelemetry spans and structured logs."""
    telemetry = TelemetryManager.get_instance()
    tracer = telemetry.get_tracer()
    logger = get_structured_logger()
    start_t = time.time()

    sanitized_args = PIIRedactor.redact_obj(kwargs)

    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("agent.name", agent_name)

        for k, v in sanitized_args.items():
            span.set_attribute(f"tool.args.{k}", str(v))

        logger.info(
            f"Tool execution started: '{tool_name}'",
            extra={
                "event": "tool.intent_start",
                "agent_name": agent_name,
                "metadata": {
                    "tool_name": tool_name,
                    "arguments": sanitized_args
                }
            }
        )

        try:
            yield span
            duration_ms = round((time.time() - start_t) * 1000, 2)
            span.set_attribute("tool.status", "SUCCESS")
            span.set_attribute("tool.duration_ms", duration_ms)

            logger.info(
                f"Tool execution completed: '{tool_name}' in {duration_ms}ms",
                extra={
                    "event": "tool.outcome_success",
                    "agent_name": agent_name,
                    "duration_ms": duration_ms,
                    "metadata": {
                        "tool_name": tool_name,
                        "status": "SUCCESS"
                    }
                }
            )
        except Exception as err:
            duration_ms = round((time.time() - start_t) * 1000, 2)
            span.record_exception(err)
            span.set_status(Status(StatusCode.ERROR, str(err)))
            span.set_attribute("tool.status", "ERROR")
            span.set_attribute("tool.duration_ms", duration_ms)

            logger.error(
                f"Tool execution failed: '{tool_name}' - {str(err)}",
                extra={
                    "event": "tool.outcome_error",
                    "agent_name": agent_name,
                    "duration_ms": duration_ms,
                    "metadata": {
                        "tool_name": tool_name,
                        "status": "ERROR",
                        "error": str(err)
                    }
                }
            )
            raise
