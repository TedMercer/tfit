# Documentation for tfit_utils class:

Author: Edward (Teddy) Mercer (TEM)  
DeLTA Lab  
Date of Creation: 4/01/2024  
Last update: 06/18/2024  

Within this folder are utilities that a user can play with to help with tfit. This includes aesthetics, visualization and speed of running the code. 

This code adds utlities to functions 

This includes:
- A GUI to hide a lot of the superfluous code within the model
- Figure presentation that can be imported

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
