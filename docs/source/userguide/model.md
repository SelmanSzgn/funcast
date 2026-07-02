# The FunCast Model

## Overview

FunCast forecasts the future trajectory of a functional process $Y$
over $[T, T+H]$ given its past trajectory over $[0, T]$ and optional
covariates $X_1, \ldots, X_p$.

## Four-step procedure

### Step 1 — Choose K

$K$ controls the richness of the future representation.
A larger $K$ allows more complex future shapes but increases
the risk of overfitting.

**Recommended range** : $K \in [4, 15]$

### Step 2 — Select $h_\ell$ via RRSS

For each covariate (including $Y$ itself), $h_\ell$ is the number
of basis functions used to represent it. It is selected automatically
by minimizing the Regularized Residual Sum of Squares (RRSS, Eq. 10).

### Step 3 — Compute $q_\ell$

The smoothing parameter $q_\ell$ controls the resolution of the
inner-product matrices :

$$q_\ell = \max(\text{degree}+1, \lfloor (1-s) \cdot h_\ell \rceil)$$

- $s = 0$ : no smoothing ($q_\ell = h_\ell$)
- $s = 1$ : maximum smoothing ($q_\ell = \text{degree}+1$)

### Step 4 — Solve the OLS problem

The coefficient vector $\hat{\beta}$ is estimated by ordinary
least squares.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `K` | int | 10 | Number of basis functions ψ for future Y |
| `s` | float | 0.5 | Smoothing coefficient ∈ [0, 1] |
| `basis_type` | str | `'bspline'` | Basis type |
| `auto_h` | bool | `True` | Auto-select hℓ via RRSS |
| `h_list` | list | `None` | Manual hℓ values |
| `degree` | int | 3 | B-spline degree |
| `rcond` | float | `None` | Pseudo-inverse threshold |

## Fitted attributes

After calling `fit()`, the following attributes are available :

| Attribute | Description |
|---|---|
| `h_values_` | Selected $h_\ell$ for each covariate |
| `q_values_` | Computed $q_\ell$ for each covariate |
| `b_hat_` | Estimated coefficient vector $\hat{\beta}$ |
| `C_list_` | Projection coefficients for each covariate |
| `J_list_` | Inner-product matrices |
| `theta_list_` | Basis matrices for each covariate |
