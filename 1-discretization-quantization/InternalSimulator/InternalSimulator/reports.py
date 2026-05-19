import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from typing import Optional


class Reports:

    def __init__(self, csv_path: str) -> None:
        self.df = pd.read_csv(csv_path)
        # Ensure reduction column is boolean (pandas may auto-convert true/false strings)
        if self.df['reduction'].dtype == object:
            self.df['reduction'] = self.df['reduction'].map({'true': True, 'false': False})  # type: ignore[arg-type]

        # Convert size columns to numeric ("unknown" -> NaN)
        for col in ['linear_size', 'lif_size']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Extract available metric headers (excluding metadata columns)
        metadata_cols = {'design', 'reduction', 'linear_size', 'lif_size', 'module'}
        self.properties = [col for col in self.df.columns if col not in metadata_cols]

        # Convert metric columns to numeric ("-" -> NaN)
        for col in self.properties:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Extract unique module names
        self.modules = sorted(self.df['module'].unique().tolist())

        # FPGA resource limits for xc7z020clg400-1 (PYNQ Z2)
        self.fpgaLimits = {
            'total_luts': 53200,
            'logic_luts': 53200,
            'lutrams': 17400,
            'ffs': 106400,
            'dsps': 220,
            'bram36': 140,
            'bram18': 280,
        }

    def plot_property(self, prop: str, modules: list[str], reduction: bool,
                      increasing: str = "linear", fix: int = 100,
                      filename: Optional[str] = None, figsize: Optional[tuple[float, float]] = None,
                      drawLimit: bool = False) -> None:
        """Plot a single property across multiple modules.

        Args:
            prop: Property to plot (e.g., 'total_luts', 'ffs')
            modules: List of modules to compare
            reduction: Whether to use reduction designs
            increasing: Which dimension varies on x-axis ("linear" or "lif", default: "linear")
            fix: Fixed value for the other dimension (default: 10)
            filename: Optional path to save the plot
            figsize: Optional tuple (width, height) in inches for figure size
        """
        # Validate increasing parameter
        increasing = increasing.lower()
        if increasing not in ["linear", "lif"]:
            raise ValueError(f"Invalid increasing value '{increasing}'. Must be 'linear' or 'lif'")

        # Validate property
        if prop not in self.properties:
            raise ValueError(f"Property '{prop}' not found. Available properties: {self.properties}")

        # Validate modules
        invalid_modules = [m for m in modules if m not in self.modules]
        if invalid_modules:
            raise ValueError(f"Invalid modules: {invalid_modules}. Available modules: {self.modules}")

        # Determine which column is x-axis and which is fixed
        x_col = "linear_size" if increasing == "linear" else "lif_size"
        fix_col = "lif_size" if increasing == "linear" else "linear_size"

        # Create plot
        fig, ax_result = plt.subplots(figsize=figsize)
        ax: Axes = ax_result  # type: ignore[assignment]

        for module in modules:
            # Filter by reduction, module, and fixed column
            filtered = self.df[
                (self.df['reduction'] == reduction) &
                (self.df['module'] == module) &
                (self.df[fix_col] == fix)
            ]

            # Sort by x-axis column for proper line plot
            filtered = filtered.sort_values(x_col)  # type: ignore[arg-type]

            # Plot the property for this module
            ax.plot(filtered[x_col], filtered[prop], marker='o', label=module)

            # Add data labels at each point
            for x, y in zip(filtered[x_col], filtered[prop]):
                if pd.notna(y):
                    label = f'{y:.3f}' if abs(y) < 1 else f'{int(y)}'
                    ax.annotate(label, (x, y), textcoords="offset points",
                               xytext=(0, 8), ha='center', fontsize=10)

        # Draw horizontal line at FPGA limit if defined for this property
        if prop in self.fpgaLimits and drawLimit:
            ax.axhline(y=self.fpgaLimits[prop], color='r', linestyle='--',
                      label=f'FPGA Limit ({self.fpgaLimits[prop]})')

        x_label = "Linear Size" if increasing == "linear" else "LIF Size"
        fix_label = "LIF" if increasing == "linear" else "Linear"
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(prop, fontsize=14)
        ax.set_title(f'{prop} - {"Reduction" if reduction else "No Reduction"} ({fix_label}={fix})', fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xscale('log')  # type: ignore[call-arg]

        if filename:
            plt.savefig(filename)
        plt.show()  # type: ignore[misc]

    def plot_module(self, reduction: bool, module: str, properties: list[str],
                    increasing: str = "linear", fix: int = 100,
                    filename: Optional[str] = None, figsize: Optional[tuple[float, float]] = None) -> None:
        """Plot multiple properties for a single module.

        Args:
            reduction: Whether to use reduction designs
            module: Module to plot (e.g., 'AcceleratorAXI', 'Neuron')
            properties: List of properties to compare
            increasing: Which dimension varies on x-axis ("linear" or "lif", default: "linear")
            fix: Fixed value for the other dimension (default: 10)
            filename: Optional path to save the plot
            figsize: Optional tuple (width, height) in inches for figure size
        """
        # Validate increasing parameter
        increasing = increasing.lower()
        if increasing not in ["linear", "lif"]:
            raise ValueError(f"Invalid increasing value '{increasing}'. Must be 'linear' or 'lif'")

        # Validate module
        if module not in self.modules:
            raise ValueError(f"Module '{module}' not found. Available modules: {self.modules}")

        # Validate properties
        invalid_properties = [p for p in properties if p not in self.properties]
        if invalid_properties:
            raise ValueError(f"Invalid properties: {invalid_properties}. Available properties: {self.properties}")

        # Determine which column is x-axis and which is fixed
        x_col = "linear_size" if increasing == "linear" else "lif_size"
        fix_col = "lif_size" if increasing == "linear" else "linear_size"

        # Filter by reduction, module, and fixed column
        filtered = self.df[
            (self.df['reduction'] == reduction) &
            (self.df['module'] == module) &
            (self.df[fix_col] == fix)
        ]

        # Sort by x-axis column for proper line plot
        filtered = filtered.sort_values(x_col)  # type: ignore[arg-type]

        # Create plot
        fig, ax_result = plt.subplots(figsize=figsize)
        ax: Axes = ax_result  # type: ignore[assignment]

        for prop in properties:
            ax.plot(filtered[x_col], filtered[prop], marker='o', label=prop)

        x_label = "Linear Size" if increasing == "linear" else "LIF Size"
        fix_label = "LIF" if increasing == "linear" else "Linear"
        ax.set_xlabel(x_label)
        ax.set_ylabel('Value')
        ax.set_title(f'{module} - {"Reduction" if reduction else "No Reduction"} ({fix_label}={fix})')
        ax.legend()
        ax.set_xscale('log')  # type: ignore[call-arg]

        if filename:
            plt.savefig(filename)
        plt.show()  # type: ignore[misc]
