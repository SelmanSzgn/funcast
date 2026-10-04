Python implementation of the FunCast functional data forecasting model.

It follows the scikit-learn API: fit / predict.

## Paper reference

Sezgin et al. (2025), *"Funcast: a forecasting model for functional data using covariates"*, under review at Journal of Statistical Planning and Inference (JSPI).

Authors : Selman Sezgin (a, b), Julien Jacques (a), Kahina Mokrani (b) and Sylvain Allio (b)

(a) ERIC, Université Lumière Lyon 2, Lyon, France

(b) Orange Research, Belfort, France

Preprint on HAL: https://hal.science/hal-05038816v1

## Installation
``
pip install funcast
``

Requires Python 3.10+

## Quick start

```python
import numpy as np
from funcast import FunCast

# y_past: (n, m1)
# y_future: (n, m1)
# covariate: (n, m1)
# t_past: (m1,)
# t_future: (m2,)
model = FunCast(K=7, s=0.8)
model.fit(y_past, y_future, t_past, t_future, covariates_past=[covariate])
y_pred = model.predict(y_past_new, covariates_past_new=[covariate_new])
# y_pred: (n_new, m2) evaluated on t_future
```

## Quick demo

Trains FunCast on synthetic curves and plots its forecasts on unseen ones.

```bash
git clone https://github.com/SelmanSzgn/funcast
cd funcast
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` on Windows PowerShell,
`source .venv/bin/activate` on macOS / Linux), then:

```bash
pip install --no-cache-dir ".[demo]"
python demo/demo.py
```

## License

MIT, see [LICENSE](LICENSE).
