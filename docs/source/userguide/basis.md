# Functional Bases

## Overview

FunCast represents functional data as linear combinations of basis
functions. Two families are available :

- **B-spline** : well suited for non-periodic signals
- **Fourier**  : well suited for periodic signals

## B-spline basis

A B-spline basis of degree $d$ with $h$ functions is built from a
knot sequence uniformly distributed over $[0, T]$.

**Key properties :**

- Local support : each function is non-zero only on a small interval
- Partition of unity : the functions sum to 1 at every point
- Smooth : $d-1$ times continuously differentiable

```python
from funcast.basis import bspline_basis
import numpy as np

t = np.linspace(0, 1, 100)
B = bspline_basis(t, n_basis=8, degree=3)
print(B.shape)   # (100, 8)
```

## Fourier basis

A Fourier basis with $h$ functions is built from cosine and sine
harmonics of increasing frequency.

**Key properties :**

- Global support : each function is non-zero everywhere
- Orthogonal : the functions are orthogonal on $[0, T]$
- Periodic : well adapted to seasonal or cyclic signals

```python
from funcast.basis import fourier_basis
import numpy as np

t = np.linspace(0, 1, 100)
B = fourier_basis(t, n_basis=7)
print(B.shape)   # (100, 7)
```

## Choosing the right basis

| Situation | Recommended basis |
|---|---|
| Signal with trend, no periodicity | B-spline |
| Seasonal or cyclic signal | Fourier |
| Unknown structure | B-spline (default) |
| Sharp local variations | B-spline |

## Using `get_basis`

The `get_basis` function is the recommended entry point :

```python
from funcast.basis import get_basis

B_bs = get_basis(t, n_basis=8, basis_type="bspline")
B_fo = get_basis(t, n_basis=8, basis_type="fourier")
```
