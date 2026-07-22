"""Isolated non-LLM offline reinforcement learning baselines."""

SUPPORTED_ALGORITHMS = ("mlp_bc", "td3_bc", "iql", "crl", "hiql")

__all__ = ["SUPPORTED_ALGORITHMS"]
