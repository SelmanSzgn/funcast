# Quick start

## Simple example

```python
import numpy as np
from funcast import FunCast

# Timestamps
t_past   = np.linspace(0, 1, 100)
t_future = np.linspace(1, 1.25, 25)

# Synthetic data
rng = np.random.default_rng(42)
n = 50
Y_past = np.sin(2 * np.pi * t_past)   + 0.1 * rng.standard_normal((n, 100))
Y_future = np.sin(2 * np.pi * t_future) + 0.1 * rng.standard_normal((n, 25))

# Training
model = FunCast(K=8, s=0.8)
model.fit(Y_past, Y_future, t_past, t_future)

# Prediction
Y_pred = model.predict(Y_past)
print(f"RMSE : {model.score(Y_past, Y_future):.4f}")
```

## Choosing the basis

```python
# B-spline (default)
model_bs = FunCast(K=8, s=0.5, basis_type="bspline")

# Fourier
model_fo = FunCast(K=8, s=0.5, basis_type="fourier")
```