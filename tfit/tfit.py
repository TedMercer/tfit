# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 13:21:16 2024

@author: TEM
"""

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from lmfit.models import GaussianModel, LorentzianModel, ConstantModel, StepModel, VoigtModel, LinearModel, QuadraticModel # type: ignore
import json
import csv
import itertools
from scipy.integrate import simps, romberg, quad # type: ignore
from scipy.signal import find_peaks # type: ignore
import matplotlib.widgets as widgets # type: ignore

#############################################################################################################################
#############################################################################################################################

class tfit:
    def __init__(self, x, y):
        '''
        Initialize the ModelFitter with x and y data.

        Input:
            x: array-like
                The x data for fitting.
                
            y: array-like
                The y data for fitting.

        Returns:
            None
        '''
        self.x = x
        self.y = y
        self.model = None
        self.params = None
        self.peak_definitions = []
        self.constant_bg = False
        self.model_created = False
        self.figures = []
#############################################################################################################################

    def manage_figures(self, fig):
        self.figures.append(fig)
        if len(self.figures) > 20:
            old_fig = self.figures.pop(0)
            plt.close(old_fig)
#############################################################################################################################

    def interp(self, start=None, stop=None, num=None, step=None, x=None):
        '''
        Interpolate data.

        Args:
            start (number, optional): The starting value of the sequence. If None,
                the minimum x value will be used.
            stop (number, optional): The end value of the sequence. If None,
                the maximum x value will be used.
            num (int, optional): Number of samples to generate.
            step (number, optional): Spacing between values. This overwrites `num`.
            x (list or array, optional): The x-coordinates at which to evaluate the interpolated values.
                This overwrites all other arguments.

        Returns:
            interpolated_fitter: tfit
                A new tfit object with interpolated data.
        '''
        if x is None:
            if start is None:
                start = min(self.x)
            if stop is None:
                stop = max(self.x)

            if step is not None:
                x = np.arange(start, stop, step=step)
            elif num is not None:
                x = np.linspace(start, stop, num=num)
            else:
                x = np.linspace(start, stop, num=len(self.x))

        y = np.interp(x, self.x, self.y)
        interpolated_fitter = tfit(x, y)
        interpolated_fitter.peak_definitions = self.peak_definitions.copy()
        interpolated_fitter.constant_bg = self.constant_bg
        interpolated_fitter.constant_bg_initial_value = getattr(self, 'constant_bg_initial_value', None)
        interpolated_fitter.constant_bg_min_value = getattr(self, 'constant_bg_min_value', None)
        interpolated_fitter.constant_bg_max_value = getattr(self, 'constant_bg_max_value', None)
        return interpolated_fitter
#############################################################################################################################

    def derivative(self, order=1):
        '''
        Returns the derivative of y-coordinates as a function of x-coordinates.

        Args:
            order (int, optional): Derivative order. Default is 1.

        Returns:
            derivative_fitter: tfit
                A new tfit object with derivative data.
        '''
        dy = np.gradient(self.y, self.x, edge_order=order)
        derivative_fitter = tfit(self.x, dy)
        derivative_fitter.peak_definitions = self.peak_definitions.copy()
        derivative_fitter.constant_bg = self.constant_bg
        derivative_fitter.constant_bg_initial_value = getattr(self, 'constant_bg_initial_value', None)
        derivative_fitter.constant_bg_min_value = getattr(self, 'constant_bg_min_value', None)
        derivative_fitter.constant_bg_max_value = getattr(self, 'constant_bg_max_value', None)
        return derivative_fitter
#############################################################################################################################

    def normalize_data(self, tomax = True, val = 1):
        '''
        Normalize the y data and return a new tfit object with normalized data.

        Returns:
            normalized_fitter: tfit
                A new tfit object with normalized data.
        '''
        if tomax == True:
            normalized_y = self.y / np.max(self.y)
            normalized_fitter = tfit(self.x, normalized_y)
            normalized_fitter.peak_definitions = self.peak_definitions.copy()
            normalized_fitter.constant_bg = self.constant_bg
            normalized_fitter.constant_bg_initial_value = getattr(self, 'constant_bg_initial_value', None)
            normalized_fitter.constant_bg_min_value = getattr(self, 'constant_bg_min_value', None)
            normalized_fitter.constant_bg_max_value = getattr(self, 'constant_bg_max_value', None)
        else: 
            normalized_y = self.y / val
            normalized_fitter = tfit(self.x, normalized_y)
            normalized_fitter.peak_definitions = self.peak_definitions.copy()
            normalized_fitter.constant_bg = self.constant_bg
            normalized_fitter.constant_bg_initial_value = getattr(self, 'constant_bg_initial_value', None)
            normalized_fitter.constant_bg_min_value = getattr(self, 'constant_bg_min_value', None)
            normalized_fitter.constant_bg_max_value = getattr(self, 'constant_bg_max_value', None)
        
        return normalized_fitter
#############################################################################################################################

    def plot(self,  definition = 'replace', xmin=-8, xmax=1, xlabel='X', ylabel='Y', title='Quick Plot', RIXS = False, norm = False, show = False):
        '''
        Quick visualization of raw data - no figure

        Input:
            xmin: float 
                The min value for the plot (Nominal RIXS value for convenience)
            xmax: float
                The max value for the plot (Nominal RIXS value for convenience)
            xlabel: string
                The label for the x-axis (given a dummy definition for quicker plotting)
            ylabel: string
                The label for the y-axis (given a dummy definition for quicker plotting)
            title: string
                The title for the plot (given a dummy definition for quicker plotting)
        '''
        if norm == False:
            if RIXS == False: 
                plt.plot(self.x, self.y, label = definition)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)

            if RIXS == True: 
                plt.plot(self.x, self.y)
                plt.xlim((xmin, xmax))
                plt.title('Quick RIXS Plot')
                plt.xlabel('Energy Loss (eV)')
                plt.ylabel('intensity (a.u.)')

        if norm == True:
            if RIXS == False: 
                plt.plot(self.x, self.y/np.max(self.y), label = definition)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)

            if RIXS == True: 
                plt.plot(self.x, self.y/np.max(self.y))
                plt.xlim((xmin, xmax))
                plt.title('Quick RIXS Plot')
                plt.xlabel('Energy Loss (eV)')
                plt.ylabel('intensity (a.u.)')
        if show == True:
            plt.show()
#############################################################################################################################

    def quick_plot(self,  definition = 'replace', xmin=-8, xmax=1, xlabel='X', ylabel='Y', title='Quick Plot', RIXS = False, norm = False):
        '''
        Quick visualization of raw data

        Input:
            xmin: float 
                The min value for the plot (Nominal RIXS value for convenience)
            xmax: float
                The max value for the plot (Nominal RIXS value for convenience)
            xlabel: string
                The label for the x-axis (given a dummy definition for quicker plotting)
            ylabel: string
                The label for the y-axis (given a dummy definition for quicker plotting)
            title: string
                The title for the plot (given a dummy definition for quicker plotting)
        '''
        fig = plt.figure()
        if norm == False:
            if RIXS == False: 
                plt.plot(self.x, self.y, label = definition)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.show()
            if RIXS == True: 
                plt.plot(self.x, self.y)
                plt.xlim((xmin, xmax))
                plt.title('Quick RIXS Plot')
                plt.xlabel('Energy Loss (eV)')
                plt.ylabel('intensity (a.u.)')
                plt.show()
        if norm == True:
            if RIXS == False: 
                plt.plot(self.x, self.y/np.max(self.y), label = definition)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.show()
            if RIXS == True: 
                plt.plot(self.x, self.y/np.max(self.y))
                plt.xlim((xmin, xmax))
                plt.title('Quick RIXS Plot')
                plt.xlabel('Energy Loss (eV)')
                plt.ylabel('intensity (a.u.)')
                plt.show()
        self.manage_figures(fig)
#############################################################################################################################

    def add_peak(self, peak_type, prefix, initial_params, bounds):
        '''
        Fundamental peak adding method to initialize a peak.

        Input:
            peak_type: string
                Peak type name based off of lmfit (i.e. 'Gaussian', 'Voigt', etc).

            prefix: string 
                Prefix of the peak (example: gaussian4_RoomTemp).

            initial_params: dictionary 
                Dictionary definition of the initial values of amplitude, sigma, and center (example: 
                {'center': -3, 'sigma': 0.4, 'amplitude': 1}).

            bounds: dictionary
                Dictionary definition of each parameter and its upper and lower bounds (example: 
                {'center': {'min': -4.5, 'max': -3.8}, 
                 'sigma': {'min': 0.035, 'max': 2}, 
                 'amplitude': {'min': 0, 'max': 0.99}}).

        Returns:
            None
        '''
        existing_peak = next((peak for peak in self.peak_definitions if peak['prefix'] == prefix), None)
        if existing_peak:
            self.peak_definitions.remove(existing_peak)
        
        self.peak_definitions.append({
            'type': peak_type,
            'prefix': prefix,
            'initial_params': initial_params,
            'bounds': bounds
        })

        self.create_model()
#############################################################################################################################

    def update_peak_parameters(self, prefix, new_initial_params=None, new_bounds=None):
        '''
        Update the initial parameters and bounds of a specified peak.

        Input:
            prefix: string
                The prefix of the peak to be updated.

            new_initial_params: dictionary, optional
                Dictionary of new initial values for the peak parameters.
                Example: {'center': -1.5, 'sigma': 0.4, 'amplitude': 1.2}

            new_bounds: dictionary, optional
                Dictionary of new bounds for the peak parameters.
                Example: {'center': {'min': -2, 'max': -1}, 'sigma': {'min': 0.1, 'max': 0.5}, 'amplitude': {'min': 0.8, 'max': 1.5}}

        Returns:
            None
        '''
        if new_initial_params:
            for param, value in new_initial_params.items():
                param_name = f'{prefix}{param}'
                if param_name in self.params:
                    self.params[param_name].set(value=value)
                else:
                    raise KeyError(f"Parameter {param_name} does not exist in the model parameters.")

        if new_bounds:
            for param, bounds in new_bounds.items():
                param_name = f'{prefix}{param}'
                if param_name in self.params:
                    if 'min' in bounds:
                        self.params[param_name].min = bounds['min']
                    if 'max' in bounds:
                        self.params[param_name].max = bounds['max']
                else:
                    raise KeyError(f"Parameter {param_name} does not exist in the model parameters.")

        print(f'Updated parameters for peak {prefix}:')
        for param in new_initial_params or []:
            print(f"{param}: {self.params[f'{prefix}{param}'].value}")
        for param in new_bounds or []:
            print(f"{param} bounds: min={self.params[f'{prefix}{param}'].min}, max={self.params[f'{prefix}{param}'].max}")
#############################################################################################################################

    def remove_peak(self, prefix):
        '''
        Remove a peak that has been added into the model.

        Input:
            prefix: string
                The prefix of the peak that you wish to remove from the model.

        Returns:
            None
        '''
        self.peak_definitions = [peak for peak in self.peak_definitions if peak['prefix'] != prefix]
        self.create_model()
#############################################################################################################################

    def add_constant_background(self, initial_value, min_value, max_value):
        '''
        Add a constant background to the model.

        Input:
            initial_value: float
                The initial value of the constant background.
                
            min_value: float
                The minimum bound for the constant background.
                
            max_value: float
                The maximum bound for the constant background.

        Returns:
            None
        '''
        self.constant_bg = True
        self.constant_bg_initial_value = initial_value
        self.constant_bg_min_value = min_value
        self.constant_bg_max_value = max_value
        self.create_model()
#############################################################################################################################

    def create_model(self):
        '''
        Create the composite model based on the added peaks and constant background.

        Returns:
            None
        '''
        self.model = None
        for peak in self.peak_definitions:
            if peak['type'] == 'Gaussian':
                component = GaussianModel(prefix=peak['prefix'])
            elif peak['type'] == 'Voigt':
                component = VoigtModel(prefix=peak['prefix'])
            elif peak['type'] == 'Lorentzian':
                component = LorentzianModel(prefix=peak['prefix'])
            else:
                raise ValueError(f"Unsupported peak type: {peak['type']}")

            if self.model is None:
                self.model = component
            else:
                self.model += component
        
        if self.constant_bg:
            self.model += ConstantModel(prefix='const_')
        
        self.params = self.model.make_params()
        self.model_created = True
#############################################################################################################################

    def set_initial_guesses_and_bounds(self):
        '''
        Set initial guesses and bounds for the model parameters.

        Returns:
            None
        '''
        for peak in self.peak_definitions:
            peak_type = peak['type']
            prefix = peak['prefix']
            initial_params = peak['initial_params']
            bounds = peak['bounds']

            if peak_type == 'Gaussian':
                component = GaussianModel(prefix=prefix)
            elif peak_type == 'Lorentzian':
                component = LorentzianModel(prefix=prefix)
            elif peak_type == 'Voigt':
                component = VoigtModel(prefix=prefix)
        
            self.params.update(component.make_params())
        
            for param, value in initial_params.items():
                self.params[f'{prefix}{param}'].set(value=value)
        
            for param, bound in bounds.items():
                self.params[f'{prefix}{param}'].set(**bound)
    
        if self.constant_bg:
            const_bg_model = ConstantModel(prefix='const_')
            self.params.update(const_bg_model.make_params())
            self.params['const_c'].set(value=self.constant_bg_initial_value, min=self.constant_bg_min_value, max=self.constant_bg_max_value)
#############################################################################################################################

    def fit_model(self):
        '''
        Fit the model to the data.

        Returns:
            result: lmfit.model.ModelResult
                The result of the fitting process.
        '''
        self.result = self.model.fit(self.y, self.params, x=self.x)
        return self.result
#############################################################################################################################

    def plot_fit(self, title, xmin, xmax, color_scheme=None, xlabel='Energy Loss (eV)', ylabel='Intensity (Norm Units)'):
        '''
        Plot the fit result along with the data and individual components.

        Input:
            title: string
                The title of the plot.
                
            xmin: float
                The minimum x value for the plot.
                
            xmax: float
                The maximum x value for the plot.
                
            color_scheme: list of strings, optional
                List of colors to use for the peaks.

        Returns:
            None
        '''
        fig = plt.figure()
        plt.plot(self.x, self.y, linestyle='--', label='Data')
        plt.plot(self.x, self.result.best_fit, 'r--', label='Fit')
        
        if color_scheme is None:
            color_scheme = ['yellow', 'purple', 'orange', 'green', 'blue', 'pink', 'red', 'brown', 'black', 'maroon']
        colors = itertools.cycle(color_scheme)
        
        for peak in self.peak_definitions:
            prefix = peak['prefix']
            color = next(colors)
            underscore_index = prefix.find('_')
            if underscore_index != -1:
                sliced_prefix = prefix[:underscore_index]
            else:
                sliced_prefix = prefix[:3]

            label = f'Peak {sliced_prefix}'
            plt.fill_between(self.x, self.result.eval_components(x=self.x)[prefix], facecolor=color, alpha=0.5, label=f'{label}')

        if self.constant_bg:
            plt.plot(self.x, self.result.eval_components(x=self.x)['const_'], label='Constant BG', color='black')

        plt.legend(fontsize=8, loc='upper left')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(xmin, xmax)
        plt.title(title)
        plt.show()
        self.manage_figures(fig)
#############################################################################################################################

    def get_fit_components(self):
        '''
        Return the x and y components of the fit.

        Returns:
            fit_components: dictionary
                Dictionary with 'x' and 'y' keys containing the fit components.
        '''
        if not self.model_created:
            raise RuntimeError("Model has not been created. Use create_model() to create the model.")
        if not self.params:
            raise RuntimeError("Parameters have not been set. Use set_initial_guesses_and_bounds() to set the parameters.")
        if not hasattr(self, 'result'):
            raise RuntimeError("Model has not been fitted. Use fit_model() to fit the model.")

        fit_components = {
            'x': self.x,
            'y': self.result.best_fit
        }
        return fit_components
#############################################################################################################################

    def get_peak_parameters(self):
        '''
        Retrieve the parameters of the peaks in the model.

        Returns:
            peak_params: dictionary
                Dictionary of peak parameters.
        '''
        peak_params = {}
        for peak in self.peak_definitions:
            prefix = peak['prefix']
            peak_params[prefix] = {}
            for param in ['center', 'sigma', 'amplitude', 'fwhm']:
                peak_params[prefix][param] = self.result.params[f'{prefix}{param}'].value

        if self.constant_bg:
            peak_params['const_'] = {'c': self.result.params['const_c'].value}

        return peak_params
#############################################################################################################################

    def shift_max_to_zero(self, x_min, x_max):
        '''
       Find the y-max within a specified x-region and shift the data so that the maximum is at x-zero.

       Input:
           x_min: float
               The minimum x value of the region to search for the y-max.

           x_max: float
               The maximum x value of the region to search for the y-max.

      Returns:
           None
      '''
       # Find the indices of the data points within the specified range
        mask = (self.x >= x_min) & (self.x <= x_max)

      # Find the x value corresponding to the maximum y value within the specified range
        if np.any(mask):
            x_region = self.x[mask]
            y_region = self.y[mask]
            max_index = np.argmax(y_region)
            x_max_pos = x_region[max_index]

          # Shift the x data so that the maximum y value is at x-zero
            shift_amount = -x_max_pos
            self.x = self.x + shift_amount

          # Optionally print the shifted data for verification
            print(f'Shifted x by {shift_amount} so that y-max is at x=0.')
        else:
            print('No data points found in the specified x range.')
        
        plt.plot(self.x, self.y)
        plt.xlabel('ARB X')
        plt.ylabel('ARB Y')
        plt.title('shifted plot')
#############################################################################################################################

    def set_max_to_nan(self, x_min, x_max):
        '''
            Find the y-max within a specified x-region and set it to NaN.

         Input:
             x_min: float
                The minimum x value of the region to search for the y-max.
            
            x_max: float
                The maximum x value of the region to search for the y-max.

        Returns:
            None
        '''
        # Find the indices of the data points within the specified range
        mask = (self.x >= x_min) & (self.x <= x_max)

        # Find the y value corresponding to the maximum y value within the specified range
        if np.any(mask):
            y_region = self.y[mask]
            max_index = np.argmax(y_region)
            max_y_value = y_region[max_index]
        
            # Set the maximum y value to NaN
            self.y[mask][max_index] = np.nan

            # Optionally print the updated y data for verification
            print(f'Set y-max value {max_y_value} in the specified x-region to NaN.')
        else:
            print('No data points found in the specified x range.')
#############################################################################################################################

    def integrate_and_sum(self, peak_num):
        '''
        Function to integrate and sum the values of multiple peaks.

        Input:
            peak_num: list of int
                List of peak indices.

        Returns:
            total_sum: float
                Total integrated and summed value.
        '''
        peak_params = self.get_peak_parameters()
        total_sum = 0.0

        for i in peak_num:
            prefix = f'g{i}_Nd_RT_LC_'
            amplitude_key = f'{prefix}amplitude'
            fwhm_key = f'{prefix}fwhm'

            
            if amplitude_key in peak_params[prefix] and fwhm_key in peak_params[prefix]:
                amplitude = peak_params[prefix][amplitude_key]
                fwhm = peak_params[prefix][fwhm_key]

                
                peak_integral = amplitude * fwhm * (2 * np.pi) ** 0.5

                
                total_sum += peak_integral
            else:
                print(f'Invalid term in selected dictionary for peak {i}')

        return total_sum
#############################################################################################################################

    def calculate_centroid(self, peak_num):
        '''
        Function to calculate the centroid of multiple peaks.

        Input:
            peak_num: list of int
                List of peak indices.

        Returns:
            centroid: tuple
                Centroid (x, y) as a tuple.
        '''
        peak_params = self.get_peak_parameters()
        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        total_amplitude = 0.0

        for i in peak_num:
            prefix = f'g{i}_Nd_RT_LC_'
            center_key = f'{prefix}center'
            amplitude_key = f'{prefix}amplitude'

            # Check if the keys exist in the dictionary
            if center_key in peak_params[prefix] and amplitude_key in peak_params[prefix]:
                center = peak_params[prefix][center_key]
                amplitude = peak_params[prefix][amplitude_key]

                weighted_sum_x += center * amplitude
                total_amplitude += amplitude
            else:
                print(f'Invalid term in selected dictionary for peak {i}')

        centroid_x = weighted_sum_x / total_amplitude if total_amplitude != 0 else 0

        weighted_sum_y = sum(peak_params[f'g{i}_Nd_RT_LC_']['center'] * peak_params[f'g{i}_Nd_RT_LC_']['amplitude'] for i in peak_num)
        total_center = sum(peak_params[f'g{i}_Nd_RT_LC_']['center'] for i in peak_num)
        centroid_y = weighted_sum_y / total_center if total_center != 0 else 0

        return centroid_x, centroid_y
#############################################################################################################################

    def shift_data(self, shift_x=0.0, shift_y=0.0):
        '''
        Shift the data set in the x or y direction.

        Input:
            shift_x: float
                The amount to shift the x data.

            shift_y: float
                The amount to shift the y data.

        Returns:
            None
        '''
        self.x = self.x + shift_x
        self.y = self.y + shift_y

        print(f'Shifted x: {self.x}')
        print(f'Shifted y: {self.y}')
#############################################################################################################################

    def save_parameters(self, filename):
        '''
        Save the current model parameters to a file.

        Input:
            filename: string
                The filename to save the parameters to.

        Returns:
            None
        '''
        with open(filename, 'w') as f:
            json.dump(self.params.valuesdict(), f)
#############################################################################################################################

    def load_parameters(self, filename):
        '''
        Load model parameters from a file.

        Input:
            filename: string
                The filename to load the parameters from.

        Returns:
            None
        '''
        with open(filename, 'r') as f:
            params_dict = json.load(f)
        for param, value in params_dict.items():
            self.params[param].set(value=value)
#############################################################################################################################

    def export_fit_results(self, filename):
        '''
        Export the fit results to a CSV file.

        Input:
            filename: string
                The filename to save the fit results to.

        Returns:
            None
        '''
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['parameter', 'value', 'stderr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for param, value in self.result.params.items():
                writer.writerow({'parameter': param, 'value': value.value, 'stderr': value.stderr})
#############################################################################################################################

    def calculate_peak_areas(self):
        '''
        Calculate and return the area under each peak.

        Returns:
            peak_areas: dictionary
                Dictionary of peak areas.
        '''
        peak_params = self.get_peak_parameters()
        peak_areas = {}
        for prefix, params in peak_params.items():
            amplitude = params.get('amplitude', 0)
            fwhm = params.get('fwhm', 0)
            area = amplitude * fwhm * np.sqrt(2 * np.pi)
            peak_areas[prefix] = area
        return peak_areas
#############################################################################################################################

    def integrate_regions_trapz(self, regions, xmin=-8, xmax=1, xlabel='Energy Loss (eV)', ylabel='Intensity'):
        '''
        Integrate the raw data over specified x-ranges using the trapezoidal rule.
    
        Input:
            regions: list of tuples
                A list of (x_min, x_max) tuples defining the integration ranges.
            xmin: float
                plot minimum value (set for nominal RIXS)
            xmax: float
                plot maximum value (set for nominal RIXS)
            xlabel: string 
                label of the xaxis (set for nominal RIXS analysis)
            ylabel: string
                label for yaxis (set fro nominal RIXS analysis)
    
        Returns:
            integrals: list of floats
                The integrals of the raw data over the specified ranges.
        '''
        integrals = []
    
        
        plt.plot(self.x, self.y, label='Data')
    
        for (x_min, x_max) in regions:
         
            mask = (self.x >= x_min) & (self.x <= x_max)

         
            integral = np.trapz(self.y[mask], self.x[mask])
            integrals.append(integral)
        
            
            plt.fill_between(self.x[mask], self.y[mask], alpha=0.5, label=f'Region [{x_min},{x_max}]')
    
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((xmin, xmax))
        plt.title('Integration of Multiple Regions -- Trap')
        plt.legend()
        plt.show()
        return integrals
#############################################################################################################################

    def get_peak_heights(self):
        '''
        Retrieve the heights of the peaks in the model.

        Returns:
            peak_heights: dictionary
                Dictionary of peak heights.
        '''
        peak_heights = {}
        for peak in self.peak_definitions:
            prefix = peak['prefix']
            amplitude = self.result.params[f'{prefix}amplitude'].value
            sigma = self.result.params[f'{prefix}sigma'].value
            height = amplitude / (sigma * np.sqrt(2 * np.pi))
            peak_heights[prefix] = height
        return peak_heights
#############################################################################################################################

    def integrate_regions_quad(self, regions, xmin=-8, xmax=1, xlabel='Energy Loss (eV)', ylabel='Intensity'):
        '''
        Integrate the raw data over specified x-ranges using adaptive quadrature.

        Input:
            regions: list of tuples
                A list of (x_min, x_max) tuples defining the integration ranges.
            xmin: float
                Plot minimum value (set for nominal RIXS).
            xmax: float
                Plot maximum value (set for nominal RIXS).
            xlabel: string 
                Label of the x-axis (set for nominal RIXS analysis).
            ylabel: string
                Label for y-axis (set for nominal RIXS analysis).

        Returns:
            integrals: list of floats
                The integrals of the raw data over the specified ranges.
        '''
        integrals = []

        
        plt.plot(self.x, self.y, label='Data')

        for (x_min, x_max) in regions:
            
            integration_func = lambda x: np.interp(x, self.x, self.y)

            
            integral, error = quad(integration_func, x_min, x_max)
            integrals.append(integral)

            
            mask = (self.x >= x_min) & (self.x <= x_max)
            plt.fill_between(self.x[mask], self.y[mask], alpha=0.5, label=f'Region [{x_min},{x_max}]')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((xmin, xmax))
        plt.title('Integration of Multiple Regions -- Quadrature')
        plt.legend()
        plt.show()

        return integrals
#############################################################################################################################

    def integrate_regions_romberg(self, regions, xmin=-8, xmax=1, xlabel='Energy Loss (eV)', ylabel='Intensity'):
        '''
        Integrate the raw data over specified x-ranges using Romberg integration.

        Input:
            regions: list of tuples
                A list of (x_min, x_max) tuples defining the integration ranges.
            xmin: float
                Plot minimum value (set for nominal RIXS).
            xmax: float
                Plot maximum value (set for nominal RIXS).
            xlabel: string 
                Label of the x-axis (set for nominal RIXS analysis).
            ylabel: string
            Label for y-axis (set for nominal RIXS analysis).

        Returns:
            integrals: list of floats
                The integrals of the raw data over the specified ranges.
        '''
        integrals = []

        
        plt.plot(self.x, self.y, label='Data')

        for (x_min, x_max) in regions:
            
            integration_func = lambda x: np.interp(x, self.x, self.y)

            
            integral = romberg(integration_func, x_min, x_max)
            integrals.append(integral)

            
            mask = (self.x >= x_min) & (self.x <= x_max)
            plt.fill_between(self.x[mask], self.y[mask], alpha=0.5, label=f'Region [{x_min},{x_max}]')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((xmin, xmax))
        plt.title('Integration of Multiple Regions -- Romberg')
        plt.legend()
        plt.show()

        return integrals
#############################################################################################################################

    def print_quick_fit_statistics(self):
        '''
        Function to print the R-squared value and the FWHM of each peak.

        Returns:
            None
        '''
        r_squared = 1 - (self.result.residual.var() / np.var(self.y))
        print(f'R-squared: {r_squared}')
        for peak in self.peak_definitions:
            prefix = peak['prefix']
            fwhm = self.result.params[f'{prefix}fwhm'].value
            print(f'FWHM of peak {prefix}: {fwhm}')
#############################################################################################################################

    def generate_full_report(self):
        """
        Generate a report of the fit results.

        Returns:
            str: The generated report.
        """
        report_lines = [
            "Fit Report for Model:",
            f"Chi-square: {self.result.chisqr}",
            f"Reduced Chi-square: {self.result.redchi}",
            f"AIC: {self.result.aic}",
            f"BIC: {self.result.bic}",
            "\nParameters:"
        ]
        for param, value in self.result.params.items():
            report_lines.append(f"{param}: {value.value} ± {value.stderr}")
        return "\n".join(report_lines)
#############################################################################################################################

    def goodness_of_fit(self):
        '''
        Calculate and return the goodness of fit statistics.

        Returns:
            gof_stats: dictionary
                Dictionary containing R-squared, adjusted R-squared, and chi-squared values.
        '''

        r_squared = 1 - (self.result.residual.var() / np.var(self.y))
        adj_r_squared = 1 - (1 - r_squared) * (len(self.y) - 1) / (len(self.y) - len(self.result.params) - 1)
        chi_squared = self.result.chisqr

        gof_stats = {
            'R-squared': r_squared,
            'Adjusted R-squared': adj_r_squared,
            'Chi-squared': chi_squared
        }
        return gof_stats
#############################################################################################################################

    def compare_models(self, other_model_result):
        '''
        Compare the current model with another model based on AIC and BIC.

        Input:
            other_model_result: lmfit.model.ModelResult
                The result of another model to compare with.

        Returns:
            None
        '''
        aic_diff = self.result.aic - other_model_result.aic
        bic_diff = self.result.bic - other_model_result.bic
        print(f'AIC difference: {aic_diff}')
        print(f'BIC difference: {bic_diff}')
#############################################################################################################################

    def add_background(self, background_type, initial_params, bounds):
        """
        Add a background to the model based on the specified type.

        Input:
            background_type: string
                Type of background (e.g., 'Constant', 'Linear', 'Quadratic').
            initial_params: dictionary
                Initial parameters for the background model.
            bounds: dictionary
                Bounds for the background parameters.
        """
        if background_type == 'Constant':
            component = ConstantModel(prefix='const_')
        elif background_type == 'Linear':
            component = LinearModel(prefix='lin_')
        elif background_type == 'Quadratic':
            component = QuadraticModel(prefix='quad_')
        else:
            raise ValueError(f"Unsupported background type: {background_type}")

        if self.model is None:
            self.model = component
        else:
            self.model += component

        self.params.update(component.make_params())
        for param, value in initial_params.items():
            self.params[f'{component.prefix}{param}'].set(value=value)
        for param, bound in bounds.items():
            self.params[f'{component.prefix}{param}'].set(**bound)
        self.constant_bg = True
        self.model_created = True
#############################################################################################################################
    
    def find_max_in_range(self, x_min, x_max):
        """
        Find the maximum value within a specified range.

        Input:
            x_min: float
                The minimum x value of the range.
            x_max: float
                The maximum x value of the range.
        Returns:
            tuple: (x, y) coordinates of the maximum value.
        """
        mask = (self.x >= x_min) & (self.x <= x_max)
        x_region = np.array(self.x[mask])
        y_region = np.array(self.y[mask])
        if len(y_region) == 0:
            raise ValueError("No data points found in the specified x range.")
        max_index = np.argmax(y_region)
        max_x = x_region[max_index]
        max_y = y_region[max_index]

        return max_x, max_y
#############################################################################################################################

    def find_min_in_range(self, x_min, x_max):
        """
        Find the minimum value within a specified range.

        Input:
            x_min: float
                The minimum x value of the range.
            x_max: float
                The maximum x value of the range.
        Returns:
            tuple: (x, y) coordinates of the minimum value.
        """
        mask = (self.x >= x_min) & (self.x <= x_max)
        x_region = np.array(self.x[mask])
        y_region = np.array(self.y[mask])
        if len(y_region) == 0:
            raise ValueError("No data points found in the specified x range.")
        min_index = np.argmin(y_region)
        min_x = x_region[min_index]
        min_y = y_region[min_index]

        return min_x, min_y
#############################################################################################################################
    
    def identify_turning_points(self):
        """
        Identify the turning points in the data.

        Returns:
            list: Array of [x, y] points corresponding to turning points.
        """
        dy = np.gradient(self.y, self.x)
        ddy = np.gradient(dy, self.x)

        peaks, _ = find_peaks(ddy)
        troughs, _ = find_peaks(-ddy)

        turning_points = np.concatenate([peaks, troughs])
        turning_points = np.sort(turning_points)

        return [(self.x[tp], self.y[tp]) for tp in turning_points]
#############################################################################################################################

    def find_differences_between_turning_points(self):
        """
        Find the x and y differences between turning points.

        Returns:
            tuple: x and y differences between turning points.
        """
        turning_points = self.identify_turning_points()
        x_diffs = [turning_points[i+1][0] - turning_points[i][0] for i in range(len(turning_points)-1)]
        y_diffs = [turning_points[i+1][1] - turning_points[i][1] for i in range(len(turning_points)-1)]

        return x_diffs, y_diffs
#############################################################################################################################

    def data_minus_fit(self):
        '''
        Subtract the fit from the data and return a new tfit object with the residuals.

        Returns:
            residual_fitter: tfit
                A new tfit object with residual data.
        '''
        if not hasattr(self, 'result'):
            raise RuntimeError("Model has not been fitted. Use fit_model() to fit the model.")

        residuals = self.y - self.result.best_fit
        residual_fitter = tfit(self.x, residuals)
        residual_fitter.peak_definitions = self.peak_definitions.copy()
        residual_fitter.constant_bg = self.constant_bg
        residual_fitter.constant_bg_initial_value = getattr(self, 'constant_bg_initial_value', None)
        residual_fitter.constant_bg_min_value = getattr(self, 'constant_bg_min_value', None)
        residual_fitter.constant_bg_max_value = getattr(self, 'constant_bg_max_value', None)
        return residual_fitter
#############################################################################################################################

    def delete(self):
        """
        Delete the object that is created.
        """
        print(f'object titled {self} is about to be deleted')
        del self
        print('its gone....')

#############################################################################################################################
######################################################DEV####################################################################
#############################################################################################################################

    def plot_second_derivative(self, bool='No'):
        """
        Plot the second derivative of the data with optional turning points.

        Input:
            bool: string
                If 'Yes', plot both peaks and troughs. If 'No', plot only troughs.
        """
        dy = np.gradient(self.y, self.x)
        ddy = np.gradient(dy, self.x)

        peaks, _ = find_peaks(ddy)
        troughs, _ = find_peaks(-ddy)

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, ddy, label='Second Derivative', color='blue')

        if bool == 'No':
            for tp in troughs:
                plt.axvline(x=self.x[tp], color='red', linestyle='--')
                plt.text(self.x[tp], ddy[tp], f'{self.x[tp]:.2f}', color='red', ha='right', va='bottom')
        elif bool == 'Yes':
            hold = np.sort(np.concatenate([peaks, troughs]))
            for tp in hold:
                plt.axvline(x=self.x[tp], color='red', linestyle='--')
                plt.text(self.x[tp], ddy[tp], f'{self.x[tp]:.2f}', color='red', ha='right', va='bottom')

        plt.title('Second Derivative with Turning Points')
        plt.xlabel('X')
        plt.ylabel('Second Derivative')
        plt.legend()
        plt.show()