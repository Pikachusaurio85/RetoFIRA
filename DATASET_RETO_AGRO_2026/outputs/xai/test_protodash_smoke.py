#!/usr/bin/env python
"""Smoke test for ProtoDash functionality."""
import numpy as np
from aix360.algorithms.protodash import ProtodashExplainer

np.random.seed(2026)
X = np.random.rand(20, 5)  # 20 parcelas, 5 dims
explainer = ProtodashExplainer()
weights, indices, values = explainer.explain(X, X, m=3, kernelType='Gaussian', sigma=1.0, optimizer='osqp')

# Verificaciones técnicas
assert len(indices) == 3, f"Esperados 3 prototipos, got {len(indices)}"
assert all(0 <= i < 20 for i in indices), "Índices fuera de rango [0,19]"
assert all(np.isfinite(weights)), "Pesos no finitos"
assert all(w >= 0 for w in weights), "Pesos negativos"

# Normalización opcional (solo si suma > 0)
w_sum = weights.sum()
weights_norm = weights / w_sum if w_sum > 0 else weights
assert np.isclose(weights_norm.sum(), 1.0) or w_sum == 0, "Normalización falló"

print("SMOKE TEST PASSED")
print(f"Índices prototipos: {indices}")
print(f"Pesos (raw): {weights}")
print(f"Pesos (normalizados): {weights_norm}")
print(f"Suma pesos raw: {w_sum:.6f}")