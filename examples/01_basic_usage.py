"""
Example 01 - FunCast usage without covariate.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from funcast import FunCast


# Synthetic data generation
# -------------------------
rng = np.random.default_rng(42)
# number of realizations
n = 60
# number of past timestamps 
m1 = 100
# number of future timestamps
m2 = 25
t_past   = np.linspace(0, 1, m1)
t_future = np.linspace(1, 1.5, m2)

freqs  = rng.uniform(0.8, 1.2, n)
phases = rng.uniform(0, np.pi / 4, n)
noise  = 0.1

Y_past = np.array([
    np.sin(2 * np.pi * freqs[i] * t_past + phases[i])
    + noise * rng.standard_normal(m1)
    for i in range(n)
])

Y_future = np.array([
    np.sin(2 * np.pi * freqs[i] * t_future + phases[i])
    + noise * rng.standard_normal(m2)
    for i in range(n)
])

# Train/test split
# ----------------
n_train = 50
Y_past_train, Y_past_test = Y_past[:n_train],   Y_past[n_train:]
Y_future_train, Y_future_test = Y_future[:n_train], Y_future[n_train:]

# Training
# --------
model = FunCast(K=8, s=0.7, basis_type="bspline")
model.fit(Y_past_train, Y_future_train, t_past, t_future)

# Inference and evaluation
# -------------------------
Y_pred = model.predict(Y_past_test)

rmse  = model.score(Y_past_test, Y_future_test, metric="rmse")
smape = model.score(Y_past_test, Y_future_test, metric="smape")

print(f"\nRMSE  : {rmse:.4f}")
print(f"SMAPE : {smape:.2f}%")

# Visualization
# -------------
mpl.rcParams['font.family'] = 'Times New Roman'
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

for i in range(min(5, len(Y_past_test))):
    ax.plot(t_past,   Y_past_test[i],   color="steelblue", alpha=0.4, lw=1)
    ax.plot(t_future, Y_future_test[i], color="steelblue", alpha=0.4, lw=1,
            linestyle="--")
    ax.plot(t_future, Y_pred[i], color="tomato", alpha=0.8, lw=1.5)

ax.axvline(x=1.0, color="gray", linestyle=":", lw=1)
ax.set_title("Test curves with FunCast predictions")
ax.set_xlabel("Timestamps")
ax.set_ylabel("Y")
ax.legend(
    handles=[
        plt.Line2D([0], [0], color="steelblue", linestyle="--", label="Target"),
        plt.Line2D([0], [0], color="tomato",    label="Prediction"),
    ],
    loc="upper right",
    fontsize=8,
)
plt.tight_layout()
plt.savefig("examples/01_basic_usage.png", bbox_inches="tight")
plt.show()
print("\nFigure saved: examples/01_basic_usage.png")
