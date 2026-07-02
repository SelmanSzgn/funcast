# Automatic Selection of $h_\ell$

## Overview

For each covariate $\ell$ (including $Y$ itself), FunCast must choose
$h_\ell$, the number of basis functions used to represent it.
This selection is done automatically by minimizing the
**Regularized Residual Sum of Squares (RRSS)**.

## How it works

FunCast evaluates the RRSS for each candidate value of $h$ in a
predefined grid, then selects the value that minimizes it :

$$\hat{h}_\ell = \arg\min_{h} \, \text{RRSS}(h)$$

A small $\hat{h}_\ell$ means the covariate is well represented by
few basis functions (smooth signal). A large $\hat{h}_\ell$ means
the covariate requires more functions (complex signal).

## Controlling the selection

### Automatic selection (default)

```python
model = FunCast(K=8, s=0.5, auto_h=True)
model.fit(Y_past, Y_future, t_past, t_future)

# Inspect selected values
print(model.h_values_)   # e.g. [12, 8]
print(model.q_values_)   # e.g. [6, 4]
```

### Manual selection

If you already know the appropriate $h_\ell$ values, you can
bypass the automatic selection :

```python
model = FunCast(K=8, s=0.5, auto_h=False, h_list=[10, 6])
model.fit(Y_past, Y_future, t_past, t_future,
          covariates_past=[X_past])
```

## Relationship between $h_\ell$ and $q_\ell$

Once $h_\ell$ is selected, $q_\ell$ is computed as :

$$q_\ell = \max(\text{degree}+1, \lfloor (1-s) \cdot h_\ell \rceil)$$

The smoothing parameter $s \in [0, 1]$ controls the compression :

| $s$ | Effect |
|---|---|
| `0.0` | No smoothing : $q_\ell = h_\ell$ |
| `0.5` | Moderate smoothing (default) |
| `1.0` | Maximum smoothing : $q_\ell = \text{degree}+1$ |
