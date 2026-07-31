"""Isolated non-LLM offline reinforcement learning baselines."""

D3RLPY_ALGORITHMS = ("mlp_bc", "td3_bc", "iql", "rebrac")
GCRL_ALGORITHMS = ("crl", "hiql")
SUPPORTED_ALGORITHMS = (*D3RLPY_ALGORITHMS, *GCRL_ALGORITHMS)

__all__ = ["D3RLPY_ALGORITHMS", "GCRL_ALGORITHMS", "SUPPORTED_ALGORITHMS"]
