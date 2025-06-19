# Thermo-Hydro-Mechanical Diffusion Simulations

This repository contains code and reports related to both version of the **PhD Challenge about Coupled Thermo-Hydro-Mechanical Simulation in an EGS**.

## Project Structure

- **Original version**: about the original version of the PhD Challenge:
  - **PhD Challenge Report**: The **PDF** version of the PhD challenge report can be found in the file [PhD Challenge Report](Original_version/PhD_Challenge_report.pdf).
  - **Extended Jupyter Report**: The **Jupyter Notebook** with an extended version of the report and more details on the results is located in [EXTENDED Challenge report](Original_version/EXTENDED_Challenge_report.ipynb).
  - **Code**: The main code for simulating the EGS system is available in the file [EGS_simulation.py](Original_version/EGS_simulation.py).
  - **Images**: This folder contains images used in the Jupyter Notebook and the LaTeX reports.
  - **Previous Computations**: The folder `Fenicsx and GMSH tries` contains earlier attempts of aborted GMSH meshes, and FEniCSx code implementations.
  - **Scikit Help Codes**: The folder `Scikit help codes` contains helper codes inspired by examples from Scikit-FEM.

- **New version**: about the new version of the PhD Challenge:
  - **PhD Challenge Report**: The **PDF** version of the PhD challenge report can be found in the file [PhD Challenge Report](New_version/PhD_Challenge_report_new.pdf).
  - **Extended Jupyter Report**: The **Jupyter Notebook** with an extended version of the report and more details on the results is located in [EXTENDED Challenge report](New_version/EXTENDED_Challenge_report_new.ipynb).
  - **Code**: The main code for simulating the EGS system is available in the file [EGS_simulation.py](New_version/EGS_simulation_new.py).
  - **Images**: This folder contains images used in the Jupyter Notebook and the LaTeX reports.

## Code Description

The code in `EGS_simulation.py` and `EGS_simulation_new.py` simulates the coupled equations and generates interactive plots that show the evolution of pressure and temperature over time. It also includes the ability to plot the evolution curves for probes.

### Key Features:

- **Interactive Plots**: The code generates interactive plots for the evolution of both temperature and pressure, with auto-stopping for each step.
- **Probes**: The code allows you to monitor the evolution of pressure and temperature at specific points, and plot these results.

### Remarks

- **Modified Equations**: The original equations provided in the subject were incorrect, and they have been modified.


### Command-Line Interface

The code can be launched very easily by the command :

```bash
python EGS_simulation.py
```
