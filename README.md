# Documentation for tfit class:

Author: Edward (Teddy) Mercer (TEM)  
DeLTA Lab  
Date of Creation: 4/01/2024  
Last update: 06/18/2024  

This is an adaptation class based on lmfit, specifically made for the analysis of x-ray spectroscopy data. It takes lmfit and simplifies the long code into short blocks to allow for efficient fits. It can be applied to most data fitting. It provides a robust framework for adding, updating, and fitting models to data with Gaussian and Voigt peaks, including options for background modeling and interactive visualization.

This code adds multiple functions that can be used on the fly when analyzing raw data.  
This includes:  
- Peak integration  
- Peak centroid  
- Region integration (multiple computational methods)  
- Retrieve the heights of the peaks in the model  

## Features
- Add and remove various peaks to the model.
- Add a constant background to the model.
- Set initial parameter guesses and bounds for the peaks.
- Fit the model to the data.
- Plot the fitted model along with the raw data.
- Calculate the parameters of the peaks in the model.
- Integrate and sum the values of multiple peaks.
- Calculate the centroid of multiple peaks.
- Integrate a specific region of the raw data using different numerical integration methods (Trapezoidal rule, Simpson's rule, Romberg integration).
- Save and load model parameters.
- Export fit results to a CSV file.
- Print and generate a report of fit statistics.

### Required Libraries
- `matplotlib.pyplot`  # For plotting
- `lmfit.models` (GaussianModel, VoigtModel, ConstantModel) # For the model components
- `numpy`  # For numerical operations
- `json`  # For saving and loading parameters in JSON format
- `csv`  # For exporting fit results to CSV
- `itertools`  # For cycling through colors in the plot
- `scipy.integrate` (simps, romberg, quad)  # For numerical integration methods
- `matplotlib.widgets` # For interactive fitting

You can install these dependencies using pip:

```sh
pip install numpy matplotlib lmfit scipy
```

### Prerequisites
- Python 3.7 or later

### IN PROGRESS
- Interactive Plotting


## Example Usage: 

Initilizing the class:
```sh
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# tem_fitter Import
sys.path.append(str(Path(r'/Users/tedmercer/Desktop/python_packages/tem_fitter')))
from tfit import tfit

# Sample x_data and y_data
x_data = np.linspace(-10, 10, 100)
y_data = np.exp(-x_data**2) + np.random.normal(0, 0.1, x_data.size)

# Initialize the class
fitter = tfit(x_data, y_data)
```

```sh
sm = .035

fitter.add_peak('Gaussian', 'g1_Nd_RT_LC_',
    {'center': -3, 'sigma': 0.4, 'amplitude': 1},
    {'center': {'min': -4.5, 'max': -3.8}, 'sigma': {'min': sm, 'max': 2}, 'amplitude': {'min': 0, 'max': 0.99}})

fitter.add_peak('Gaussian', 'g2_Nd_RT_LC_',
    {'center': -2, 'sigma': 0.4, 'amplitude': 1},
    {'center': {'min': -1.7, 'max': -1.5}, 'sigma': {'min': sm, 'max': 0.3}, 'amplitude': {'min': 0}})

fitter.add_peak('Voigt', 'v1_Nd_RT_LC_',
    {'center': 0, 'sigma': 0.4, 'amplitude': 1},
    {'center': {'min': -0.2, 'max': 0.2}, 'sigma': {'min': sm, 'max': 2}, 'amplitude': {'min': 0, 'max': 0.99}})

fitter.add_constant_background(initial_value=0.00325, min_value=0.003, max_value=0.0035)
fitter.create_model()
fitter.set_initial_guesses_and_bounds()
result = fitter.fit_model()
fitter.plot_fit('Nd Resonant', -8, 1)
fitter.goodness_of_fit()
fitter.print_quick_fit_statistics()


---------------- Say you want to integrate large data sets --------------------
elements_RT = ['La', 'Pr', 'Nd', 'Er', 'Sm']
results_RT = [None] * len(elements_RT)
plt.figure()
for i, element in enumerate(elements_RT):
    x_data = RLC[f'{element}_RT_x']
    y_data = RLC[f'{element}_RT_y']
    fitter = tfit(x_data, y_data)
    
    region = [(-2, -1.25)]

    results_RT[i] = fitter.integrate_regions_trapz(region)
    print(f'integration done of region {region} it has been added to element {i}')
    
result_2d_RT = np.column_stack((elements_RT, results_RT))

---------------- Say you want to use the GUI --------------------
import tkinter as tk
from tfit import tfit
from FitGUI import FitGUI

# Sample x_data and y_data
x_data = np.linspace(-10, 10, 100)
y_data = np.exp(-x_data**2) + np.random.normal(0, 0.1, x_data.size)

# Initialize tfit
fitter = tfit(x_data, y_data)

# Initialize FitGUI
root = tk.Tk()
app = FitGUI(root, fitter=fitter, name='Sample Fit')
root.mainloop()

```
