"""
Demonstration script of FunCast training and inference.

Main steps:
    1) synthetic functional data generation with two covariates using Gaussian
    basis and fixed functional parameters,
    2) FunCast training and inference on test set,
    3) plot predictions and print test metrics.

References
----------
Sezgin, S. and al. (2025). "FunCast: a forecasting model for functional data
using covariates". https://hal.science/hal-05038816v1
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score

import funcast
from funcast import FunCast


np.random.seed(42)

# number of observations
n = 1000
# number of timestamps
m = 200
# past window ratio
tau = 0.7
# training data ratio
train_frac = 0.8
# number of basis functions for covariates
nbas = 10
# number of past timestamps
m1 = int(tau * m)
# all timestamps
t_all = np.linspace(0, 1, m)
# past timestamps
t_past = t_all[:m1]
# future timestamps
t_future = t_all[m1:]
# number of future timestamps
m2 = len(t_future)
# number of training data
n_train = int(n * train_frac)
# modes of Gaussian basis functions
mu = np.linspace(0, 1, nbas)


def get_gauss(m, nbas, t_all, mu):
    """Construct the Gaussian basis matrix."""
    gauss = np.zeros((m, nbas))
    for j in range(m):
        for k in range(nbas):
            gauss[j, k] = norm.pdf(t_all[j], loc=mu[k], scale=0.05)
    return gauss


def gen_covar(n, m, nbas, gauss):
    """Generate the observations of one covariate."""
    x = np.zeros((n, m))
    for i in range(n):
        coefs = norm.rvs(loc=0, scale=1, size=nbas)
        for j in range(m):
            x[i, j] = np.dot(coefs, gauss[j, :])
    return x


gauss = get_gauss(m, nbas, t_all, mu)
x1 = gen_covar(n, m, nbas, gauss)
x2 = gen_covar(n, m, nbas, gauss)
# functional parameters
gam1 = np.cos(2 * np.pi * t_all / 0.2)
gam2 = np.sin(2 * np.pi * t_all / 0.2)

y = np.zeros((n, m))
for i in range(n):
    # functional concurrent model
    y[i] = gam1*x1[i] + gam2*x2[i]
# plot functional parameters and covariates
figs, axs = plt.subplots(nrows=2, ncols=2)
for i in range(10):
    axs[0, 0].plot(t_all, x1[i, :], color="blue")
    axs[0, 0].set_xlabel("Timestamp")
    axs[0, 0].set_title("X_1(t) covariate")

    axs[0, 1].plot(t_all, x2[i, :], color="green")
    axs[0, 1].set_xlabel("Timestamp")
    axs[0, 1].set_title("X_2(t) covariate")

    axs[1, 0].plot(t_all, gam1, color="magenta")
    axs[1, 0].set_xlabel("Timestamp")
    axs[1, 0].set_title("gamma_1(t) parameter")

    axs[1, 1].plot(t_all, gam2, color="magenta")
    axs[1, 1].set_xlabel("Timestamp")
    axs[1, 1].set_title("gamma_2(t) parameter")
plt.tight_layout()
plt.show()
# plot response Y(t)
figs, ax = plt.subplots()
for i in range(5):
    ax.plot(t_all, y[i], color="brown")
    ax.set_xlabel("Timestamp")
    ax.set_title("Y(t) response function")
plt.show()

y_past_train = y[:n_train, :m1]
y_future_train = y[:n_train, m1:]
y_past_test = y[n_train:, :m1]
y_future_test = y[n_train:, m1:]
covariates_past_train = [x1[:n_train, :m1], x2[:n_train, :m1]]
covariates_past_test = [x1[n_train:, :m1], x2[n_train:, :m1]]

# funcast training and inference
model = FunCast(K=7, s=0.8)
model.fit(
    y_past_train,
    y_future_train,
    t_past,
    t_future,
    covariates_past=covariates_past_train
)
y_pred = model.predict(
    y_past_test,
    covariates_past_new=covariates_past_test
)
# plot funcast inference
fig, ax = plt.subplots(nrows=3, ncols=3)
for l, a in enumerate(ax.flatten()):
    a.plot(t_all, np.concatenate((y_past_test[l], y_future_test[l])),
            color="brown", alpha=0.7, label="Y(t)", ls="--")
    a.plot(t_future, y_pred[l], color="red", label="FunCast pred.")
    a.set_xlabel("Timestamp")
    a.legend(fontsize=8, loc="upper left")
plt.suptitle("FunCast inference vs. true future of Y")
plt.tight_layout()
plt.show()
# compute and print test metrics
mae = np.mean(np.abs(y_pred.flatten() - y_future_test.flatten()))
nmae = mae / np.std(y_future_test.flatten())
rmse = np.sqrt(np.mean((y_pred.flatten() - y_future_test.flatten()) ** 2))
r2 = r2_score(y_future_test.flatten(), y_pred.flatten())
print("Metrics on test set:")
print(f"RMSE : {rmse:.3g}")
print(f"R²   : {r2:.3g}")
print("Demo finished.")




