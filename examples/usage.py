"""
Example of FunCast usage with covariate.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from funcast import FunCast

# example parameters
rng = np.random.default_rng(0)
# Number of realizations
n = 80
# Number of past timestamps
m1 = 100
# Number of future timestamps
m2 = 20
# total interval length
interval = 1
# past interval ratio (between 0 and 1)
tau = 0.7
t_past = np.linspace(0, tau * interval, m1)
t_future = np.linspace(tau * interval, interval, m2)
# noise
sigma = 0.3

# synthetic data generation
temp_mean = rng.uniform(10, 25, n)
temp_amp = rng.uniform(3, 8, n)

X_past = np.array(
    [
        temp_mean[i]
        + temp_amp[i] * np.cos(2 * np.pi * t_past)
        + sigma * rng.standard_normal(m1)
        for i in range(n)
    ]
)

beta = rng.uniform(-0.3, -0.1, n)

Y_past = np.array(
    [
        50
        + beta[i] * X_past[i]
        + np.sin(2 * np.pi * t_past)
        + sigma * rng.standard_normal(m1)
        for i in range(n)
    ]
)

Y_future = np.array(
    [
        50
        + beta[i] * (temp_mean[i] + temp_amp[i] * np.cos(2 * np.pi * t_future))
        + np.sin(2 * np.pi * t_future)
        + sigma * rng.standard_normal(m2)
        for i in range(n)
    ]
)

# train/test split
n_train = 65
Y_past_train, Y_past_test = Y_past[:n_train], Y_past[n_train:]
Y_future_train, Y_future_test = Y_future[:n_train], Y_future[n_train:]
X_past_train, X_past_test = X_past[:n_train], X_past[n_train:]

# training et inference
model = FunCast(K=8, s=0.5)
model.fit(
    Y_past_train,
    Y_future_train,
    t_past,
    t_future,
    covariates_past=[X_past_train],
)

Y_pred = model.predict(Y_past_test, covariates_past_new=[X_past_test])

rmse = np.sqrt(np.mean((Y_pred.flatten() - Y_future_test.flatten()) ** 2))

print(f"\nRMSE  : {rmse:.4f}")

# visualization
mpl.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
for i in range(min(5, len(Y_past_test))):
    ax.plot(t_past, Y_past_test[i], color="black", alpha=0.4, lw=1)
    ax.plot(
        t_future,
        Y_future_test[i],
        color="black",
        alpha=0.4,
        lw=1,
        linestyle="--",
    )
    ax.plot(t_future, Y_pred[i], color="tomato", alpha=0.8, lw=1.5)

ax.axvline(x=1.0, color="gray", linestyle=":", lw=1)
ax.set_title("Test curves with FunCast predictions")
ax.set_xlabel("Timestamps")
ax.set_ylabel("Y")
ax.legend(
    handles=[
        plt.Line2D([0], [0], color="black", linestyle="--", label="Target"),
        plt.Line2D([0], [0], color="tomato", label="Prediction"),
    ],
    loc="upper right",
    fontsize=8,
)

plt.tight_layout()
plt.savefig("examples/usage.png", bbox_inches="tight")
plt.show()
print("\nFigure saved: examples/usage.png")
