Python implementation of the FunCast functional data forecasting model.

## Paper reference

Sezgin et al. (2025), *"Funcast: a forecasting model for functional data using covariates"*, under review at Journal of Statistical Planning and Inference (JSPI).

Authors : Selman Sezgin (a, b), Julien Jacques (a), Kahina Mokrani (b) and Sylvain Allio (b)

(a) ERIC, Université Lumière Lyon 2, Lyon, France

(b) Orange Research, Belfort, France

## Installation
``
pip install funcast
``

## Quick start
```
import numpy as np
from funcast import FunCast

n, m1, m2 = 50, 100, 20
t_past   = np.linspace(0, 1, m1)
t_future = np.linspace(1, 1.2, m2)
Y_past   = np.random.randn(n, m1)
Y_future = np.random.randn(n, m2)

model = FunCast(K=8, s=0.5)
model.fit(Y_past, Y_future, t_past, t_future)
Y_pred = model.predict(Y_past)
```