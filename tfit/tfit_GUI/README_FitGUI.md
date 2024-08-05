# Documentation for FitGUI class:

Author: Edward (Teddy) Mercer (TEM)  
DeLTA Lab  
Date of Creation: 6/16/2024  
Last update: 06/18/2024  

This is a GUI for introducing fitting utilizing tfit. It allows the user to adjust peak parameters within a GUI and see the changes to the model live. Additionally, after each plot, the chosen parameters are printed for the user's documentation.

This code includes a user interface for tfit functionality (an adaptation of lmfit).

## Features
- Add and remove peaks with the click of a button
- Adjust the parameters within the interface
- Update the model with each run
- Change axis titles

### Required Libraries
- `import tkinter as tk`
- `from tkinter import messagebox`
- `from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg`
- `import matplotlib.pyplot as plt`
- `import numpy as np`
- `from lmfit.models import GaussianModel, LorentzianModel, ConstantModel, StepModel, VoigtModel, LinearModel`
- `import sys`
- `from pathlib import Path`

You can install these dependencies using pip:

```sh
pip install numpy matplotlib lmfit
```

### Prerequisites
- Python 3.7 or later

### IN PROGRESS
- A lot of things

## Example Usage: 

Initilizing the class:
```sh
import numpy as np
import matplotlib.pyplot as plt
#tem_fitter Import
sys.path.append(str(Path(r'/Users/tedmercer/Desktop/python_packages/tfit')))
from tfit import tfit
from FitGUI import FitGUI

# Sample x_data and y_data
x_data = np.linspace(-10, 10, 100)
y_data = np.exp(-x_data**2) + np.random.normal(0, 0.1, x_data.size)

#tfit
randg = tfit(x_data, y_data)

```

Example of processing the data into the application

```sh
root = tk.Tk()
app = FitGUI(root, fitter = randg, name='randg')
root.mainloop()
```