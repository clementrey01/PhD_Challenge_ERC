# Thermo-Hydro-Mechanical Diffusion Simulations

This repository contains code and reports related to the **PhD Challenge about Coupled Thermo-Hydro-Mechanical Simulation in an EGS**.

## Project Structure

- **PhD Challenge Report**: The **PDF** version of the PhD challenge report can be found in the file [PhD Challenge Report](PhD_Challenge_report.pdf).
- **Extended Jupyter Report**: The **Jupyter Notebook** with an extended version of the report and more details on the results is located in [EXTENDED Challenge report](EXTENDED_Challenge_report.ipynb).
- **Code**: The main code for simulating the EGS system is available in the file [EGS_simulation.py](EGS_simulation.py).
- **Images**: This folder contains images used in the Jupyter Notebook and the LaTeX reports.
- **Previous Computations**: The folder `Fenicsx and GMSH tries` contains earlier attempts of aborted GMSH meshes, and FEniCSx code implementations.
- **Scikit Help Codes**: The folder `Scikit help codes` contains helper codes inspired by examples from Scikit-FEM.

## Code Description

The code in `EGS_simulation.py` simulates the coupled equations and generates interactive plots that show the evolution of pressure and temperature over time. It also includes the ability to plot the evolution curves for probes.

### Key Features:

- **Interactive Plots**: The code generates interactive plots for the evolution of both temperature and pressure, with auto-stopping for each step.
- **Probes**: The code allows you to monitor the evolution of pressure and temperature at specific points, and plot these results.

### Remarks

- **Modified Equations**: The original equations provided in the subject were incorrect, and they have been modified.


### Command-Line Interface

The code accepts several command-line arguments for controlling the simulation and the visualization.

#### **Commands**

```bash
python EGS_simulation.py [OPTIONS]
```

##### ** Arguments**

- `-f`, `--field` : Choose the field to visualize, between :
  - `temperature` (default)
  - `pressure`
- `--dual` : Show both fields (temperature and pressure) side by side.
