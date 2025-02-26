# Nelson-Siegel Curve Estimation

This repository provides a Python implementation for estimating the Nelson-Siegel yield curve. The Nelson-Siegel model is widely used to fit the term structure of interest rates, offering a smooth yield curve representation based on four key parameters: Intercept, Slope, Curvature, and Lambda.

## Features

- **Nelson-Siegel Curve Fitting**: Estimate the parameters of the yield curve using historical bond yield data.
- **Curve Generation**: Generate the estimated yield curve for specific maturities.
- **Visualization**:
  - Historical evolution of Nelson-Siegel parameters with a 5-week moving average.
  - Comparison between observed bond yields and the fitted Nelson-Siegel curve.

## Installation

### Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `scipy`
- `tqdm`
- `matplotlib`

You can install the required dependencies using:

```sh
pip install numpy pandas scipy tqdm matplotlib
```

## Usage

### 1. Initialize the Nelson-Siegel Estimator

```python
import numpy as np
from nsc import nsc

maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])  # Example maturities in years
ns_model = nsc(maturities)
```

### 2. Fit the Model to Bond Yield Data

```python
import pandas as pd

# Example DataFrame with dates as index and maturities as columns
yields = pd.DataFrame(
    {
        1: [3.5, 3.6, 3.7],
        2: [3.6, 3.7, 3.8],
        3: [3.7, 3.8, 3.9],
        5: [3.9, 4.0, 4.1],
        7: [4.1, 4.2, 4.3],
        10: [4.3, 4.4, 4.5],
        20: [4.5, 4.6, 4.7],
        30: [4.7, 4.8, 4.9],
    },
    index=pd.date_range("2023-01-01", periods=3, freq="W"),
)

# Fit the Nelson-Siegel model
ns_params = ns_model.fit(yields)
print(ns_params)
```

### 3. Generate the Yield Curve for Specific Maturities

```python
curve_maturities = np.linspace(0.5, 30, 60)  # Generate curve for a smooth range of maturities
ns_curve = ns_model.generate_curve(curve_maturities)
print(ns_curve)
```

### 4. Plot Historical Evolution of Nelson-Siegel Parameters

```python
from nsc import nsc_histo

nsc_histo(ns_params)
```

### 5. Compare Observed Yields with the Nelson-Siegel Curve

```python
from nsc import nsc_comps

# Example of a specific date's yield curve
date = ns_params.index[-1]
observed_yields = yields.loc[date]
ns_estimated = ns_curve.loc[date]
nsc_comps(observed_yields, maturities, ns_estimated, curve_maturities, ns_params.loc[date])
```

## Author

Developed by Mattéo Bernard.

