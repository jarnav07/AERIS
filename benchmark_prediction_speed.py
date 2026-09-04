#!/usr/bin/env python3
"""Benchmark prediction speed for each model type."""

import timeit
import numpy as np
from src.airfoil_ml.models import make_models, ModelConfig

# Feature dimensions from training.py
N_FEATURES = 26  # Kulfan (18) + geometry (6) + flow (alpha, Re, mach, n_crit, xtr_upper, xtr_lower = 6) + padding
N_TARGETS = 3   # CL, CD, CM

# Generate synthetic training data
rng = np.random.RandomState(42)
X_train = rng.randn(500, N_FEATURES)
y_train = rng.randn(500, N_TARGETS)
X_test = rng.randn(1, N_FEATURES)  # Single airfoil prediction

models = make_models(ModelConfig())

print("Model Prediction Speed Benchmark")
print("=" * 60)
print(f"Input features: {N_FEATURES}")
print(f"Output targets: {N_TARGETS}")
print()

results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("done.")

    # Warmup
    _ = model.predict(X_test)

    # Time 1000 single predictions
    time_per_prediction = timeit.timeit(
        lambda: model.predict(X_test),
        number=1000
    ) / 1000

    results[model_name] = time_per_prediction
    print(f"  → {time_per_prediction * 1000:.4f} ms per prediction")

print()
print("=" * 60)
print("Ranking by speed:")
for model_name, time_ms in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {model_name:15s}: {time_ms * 1000:.4f} ms")

fastest = min(results.items(), key=lambda x: x[1])
print()
print(f"Fastest model: {fastest[0]} at {fastest[1] * 1000:.4f} ms per prediction")
print(f"             = {fastest[1] * 1000000:.1f} microseconds per airfoil")
