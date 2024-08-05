# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:21:16 2024

@author: TEM
"""

import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, ConstantModel, VoigtModel

class FitGUI:
    def __init__(self, root, fitter, name='Fit Model'):
        self.root = root
        self.root.title("Peak Manager")
        self.fitter = fitter

        self.peak_entries = []

        # Title of the fitter model
        tk.Label(root, text=f"{name} Model").grid(row=0, column=4, columnspan=2, padx=5, pady=5)
        tk.Label(root, text="Teddy Mercer-Δ Lab NU").grid(row=0, column=8, padx=5, pady=5)

        # Axis labels and title
        tk.Label(root, text="X-axis Label").grid(row=1, column=0, padx=5, pady=5)
        self.xlabel_entry = tk.Entry(root)
        self.xlabel_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(root, text="Y-axis Label").grid(row=1, column=2, padx=5, pady=5)
        self.ylabel_entry = tk.Entry(root)
        self.ylabel_entry.grid(row=1, column=3, padx=5, pady=5)

        tk.Label(root, text="Title").grid(row=1, column=4, padx=5, pady=5)
        self.title_entry = tk.Entry(root)
        self.title_entry.grid(row=1, column=5, padx=5, pady=5)

        labels = [
            "Prefix", "Center", "Sigma", "Amplitude",
            "Center Min", "Center Max",
            "Sigma Min", "Sigma Max",
            "Amplitude Min", "Amplitude Max"
        ]

        # Plus and minus buttons
        self.add_peak_button = tk.Button(root, text="+", command=self.add_peak_row)
        self.add_peak_button.grid(row=2, column=4, padx=5, pady=5)

        self.remove_peak_button = tk.Button(root, text="-", command=self.remove_peak_row)
        self.remove_peak_button.grid(row=2, column=5, padx=5, pady=5)

        self.submit_button = tk.Button(root, text="Submit", command=self.submit_peaks)
        self.submit_button.grid(row=2, column=6, padx=5, pady=5)

        # Column labels
        for i, label in enumerate(labels):
            tk.Label(root, text=label).grid(row=3, column=i, padx=5, pady=5)

        self.add_peak_row()  # Initialize with one peak row

    def add_peak_row(self):
        row = len(self.peak_entries) + 5
        peak_data = {
            'prefix': tk.StringVar(value=f'g{row-5}_'),
            'center': tk.DoubleVar(value=0),
            'sigma': tk.DoubleVar(value=1),
            'amplitude': tk.DoubleVar(value=1),
            'center_min': tk.DoubleVar(value=-1),
            'center_max': tk.DoubleVar(value=1),
            'sigma_min': tk.DoubleVar(value=0.5),
            'sigma_max': tk.DoubleVar(value=2),
            'amplitude_min': tk.DoubleVar(value=0),
            'amplitude_max': tk.DoubleVar(value=2)
        }
        self.peak_entries.append(peak_data)

        tk.Entry(self.root, textvariable=peak_data['prefix'], width=10).grid(row=row, column=0, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['center'], width=10).grid(row=row, column=1, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['sigma'], width=10).grid(row=row, column=2, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['amplitude'], width=10).grid(row=row, column=3, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['center_min'], width=10).grid(row=row, column=4, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['center_max'], width=10).grid(row=row, column=5, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['sigma_min'], width=10).grid(row=row, column=6, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['sigma_max'], width=10).grid(row=row, column=7, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['amplitude_min'], width=10).grid(row=row, column=8, padx=5, pady=5)
        tk.Entry(self.root, textvariable=peak_data['amplitude_max'], width=10).grid(row=row, column=9, padx=5, pady=5)

    def remove_peak_row(self):
        if self.peak_entries:
            for widget in self.root.grid_slaves(row=len(self.peak_entries) + 4):
                widget.grid_forget()
            self.peak_entries.pop()

    def submit_peaks(self):
        self.fitter.peak_definitions = []
        for i, peak_data in enumerate(self.peak_entries):
            prefix = peak_data['prefix'].get()
            peak_type = 'Gaussian'
            if prefix.startswith('l'):
                peak_type = 'Lorentzian'
            elif prefix.startswith('v'):
                peak_type = 'Voigt'

            self.fitter.add_peak(
                peak_type,
                prefix,
                {
                    'center': peak_data['center'].get(),
                    'sigma': peak_data['sigma'].get(),
                    'amplitude': peak_data['amplitude'].get()
                },
                {
                    'center': {'min': peak_data['center_min'].get(), 'max': peak_data['center_max'].get()},
                    'sigma': {'min': peak_data['sigma_min'].get(), 'max': peak_data['sigma_max'].get()},
                    'amplitude': {'min': peak_data['amplitude_min'].get(), 'max': peak_data['amplitude_max'].get()}
                }
            )
            print(f"update_params_p{i+1} = {{'center': {peak_data['center'].get()}, 'sigma': {peak_data['sigma'].get()}, 'amplitude': {peak_data['amplitude'].get()}}}")
            print(f"update_bounds_p{i+1} = {{'center': {{'min': {peak_data['center_min'].get()}, 'max': {peak_data['center_max'].get()}}}, "
                  f"'sigma': {{'min': {peak_data['sigma_min'].get()}, 'max': {peak_data['sigma_max'].get()}}}, "
                  f"'amplitude': {{'min': {peak_data['amplitude_min'].get()}, 'max': {peak_data['amplitude_max'].get()}}}}}")

        self.fitter.create_model()
        self.fitter.set_initial_guesses_and_bounds()
        result = self.fitter.fit_model()

        self.plot_fit()

    def plot_fit(self):
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Fit Plot")

        fig, ax = plt.subplots()
        ax.plot(self.fitter.x_data, self.fitter.y_data, 'b', label='Data')
        ax.plot(self.fitter.x_data, self.fitter.result.best_fit, 'r--', label='Fit')

        # Print the keys of eval_components to debug the issue
        eval_components = self.fitter.result.eval_components(x=self.fitter.x_data)
        print("Eval components keys:", eval_components.keys())

        colors = ['yellow', 'purple', 'orange', 'green', 'blue', 'pink', 'red', 'brown', 'black', 'maroon']
        for i, peak in enumerate(self.fitter.peak_definitions):
            prefix = peak['prefix']
            color = colors[i % len(colors)]
            if prefix in eval_components:
                ax.fill_between(self.fitter.x_data, 0, eval_components[prefix], facecolor=color, alpha=0.5, label=f'Peak {prefix}')
            else:
                print(f"Warning: {prefix} not found in eval_components")

        r_squared = 1 - (self.fitter.result.residual.var() / np.var(self.fitter.y_data))
        ax.text(0.05, 0.95, f'R²: {r_squared:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

        ax.legend()

        xlabel = self.xlabel_entry.get() if self.xlabel_entry.get() else "X-axis"
        ylabel = self.ylabel_entry.get() if self.ylabel_entry.get() else "Y-axis"
        title = self.title_entry.get() if self.title_entry.get() else "Fit Plot"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)