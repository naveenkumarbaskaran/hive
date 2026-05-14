"""Backward-compatibility shim — real implementation lives in hive/llm_client.py."""
from hive.llm_client import *  # noqa: F401,F403
from hive.llm_client import llm, LLMClient, LLMResponse, ModelTier  # noqa: F401
