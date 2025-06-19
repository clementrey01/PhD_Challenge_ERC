"""
Coupled Heat and Pressure Diffusion Equations (Uncoupled)

This solves two independent diffusion equations simultaneously:
- ∂T/∂t = κ_T ∇²T  (heat equation)
- ∂p/∂t = κ_p ∇²p  (pressure equation)

Both equations are solved on the same mesh but evolve independently.
"""

from math import ceil
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import block_diag

from skfem import MeshQuad, Basis, ElementQuad2, asm, penalize
from skfem.models.poisson import laplace, mass


# Physical parameters
halfwidth = np.array([2., 3.])
ncells = 2**3
thermal_diffusivity = 5.0
pressure_diffusivity = 3.0  # Different diffusivity for pressure

# Create mesh
mesh = MeshQuad.init_tensor(
    np.linspace(-1, 1, 2*ncells) * halfwidth[0],
    np.linspace(-1, 1, 2*ncells * ceil(halfwidth[1] / halfwidth[0])) * halfwidth[1])

# Create basis functions (same for both fields)
element = ElementQuad2()
basisT = Basis(mesh, element)  # Temperature basis
basisp = Basis(mesh, element)  # Pressure basis

print(f"Number of DOFs per field: {basisT.N}")
print(f"Total DOFs: {2 * basisT.N}")

# Assemble matrices for temperature
L_T = thermal_diffusivity * asm(laplace, basisT)
M_T = asm(mass, basisT)

# Assemble matrices for pressure  
L_p = pressure_diffusivity * asm(laplace, basisp)
M_p = asm(mass, basisp)

# Time stepping parameters
dt = 0.01
print('dt =', dt)
theta = 0.5  # Crank-Nicolson

# Apply boundary conditions (homogeneous Dirichlet: u=0 on boundary)
L0_T, M0_T = penalize(L_T, M_T, D=basisT.get_dofs())
L0_p, M0_p = penalize(L_p, M_p, D=basisp.get_dofs())

# Form system matrices for each field
A_T = M0_T + theta * L0_T * dt      # Temperature LHS matrix
B_T = M0_T - (1 - theta) * L0_T * dt # Temperature RHS matrix

A_p = M0_p + theta * L0_p * dt      # Pressure LHS matrix  
B_p = M0_p - (1 - theta) * L0_p * dt # Pressure RHS matrix

# Create block matrices for the coupled system
# Structure: [T_nodes, p_nodes] as unknowns
A_block = block_diag([A_T, A_p], format='csc')
B_block = block_diag([B_T, B_p], format='csc')

print(f"Block matrix shape: {A_block.shape}")

# Factor the system matrix (done once since A doesn't change)
backsolve = splu(A_block).solve

# Initial conditions (same cosine distribution for both fields)
u_T_init = np.cos(np.pi * basisT.doflocs / 2 / halfwidth[:, None]).prod(0)
u_p_init = np.cos(np.pi * basisp.doflocs / 2 / halfwidth[:, None]).prod(0)

# Combined initial state vector: [T_values, p_values]
u_init = np.concatenate([u_T_init, u_p_init])

print(f"Initial state vector size: {len(u_init)}")

# Analytical solutions (for validation)
def exact_T(t: float) -> np.ndarray:
    """Exact solution for temperature field"""
    decay_rate = thermal_diffusivity * np.pi**2 / 4 * sum(halfwidth**-2)
    return np.exp(-decay_rate * t) * u_T_init

def exact_p(t: float) -> np.ndarray:
    """Exact solution for pressure field"""
    decay_rate = pressure_diffusivity * np.pi**2 / 4 * sum(halfwidth**-2)
    return np.exp(-decay_rate * t) * u_p_init

def exact_combined(t: float) -> np.ndarray:
    """Combined exact solution vector"""
    return np.concatenate([exact_T(t), exact_p(t)])

# Time evolution generator
def evolve(t: float, u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    """
    Time evolution using implicit Crank-Nicolson scheme
    
    At each step:
    1. Compute RHS = B_block @ u_current
    2. Solve A_block @ u_new = RHS
    3. Return new time and solution
    """
    while np.linalg.norm(u, np.inf) > 2**-4:  # Continue until solution decays
        rhs = B_block @ u
        u_new = backsolve(rhs)
        t_new = t + dt
        yield t_new, u_new
        u = u_new

# Utility functions for extracting individual fields
def extract_fields(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract temperature and pressure from combined solution vector"""
    n_dofs = len(u) // 2
    u_T = u[:n_dofs]
    u_p = u[n_dofs:]
    return u_T, u_p

# Probe points for monitoring solution at center
probe_T = basisT.probes(np.zeros((mesh.dim(), 1)))
probe_p = basisp.probes(np.zeros((mesh.dim(), 1)))

def probe_center(u: np.ndarray) -> Tuple[float, float]:
    """Evaluate both fields at domain center"""
    u_T, u_p = extract_fields(u)
    T_center = (probe_T @ u_T)[0]
    p_center = (probe_p @ u_p)[0]
    return T_center, p_center


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from skfem.visuals.matplotlib import plot

    parser = ArgumentParser(description='Heat and pressure diffusion in rectangle')
    parser.add_argument('-g', '--gif', action='store_true', 
                        help='write animated GIF')
    parser.add_argument('-f', '--field', choices=['temperature', 'pressure'], 
                        default='temperature', help='which field to visualize')
    args = parser.parse_args()

    # Setup visualization
    u_T_init_viz, u_p_init_viz = extract_fields(u_init)
    
    if args.field == 'temperature':
        field_data = u_T_init_viz[basisT.nodal_dofs.flatten()]
        field_name = 'Temperature'
        colormap = 'hot'
    else:
        field_data = u_p_init_viz[basisp.nodal_dofs.flatten()]  
        field_name = 'Pressure'
        colormap = 'viridis'

    ax = plot(mesh, field_data, shading='gouraud', cmap=colormap)
    title = ax.set_title(f'{field_name} at t = 0.00')
    field_plot = ax.get_children()[0]
    fig = ax.get_figure()
    fig.colorbar(field_plot, label=field_name)

    # Animation update function
    def update(event):
        t, u = event
        u_T, u_p = extract_fields(u)
        
        # Get exact solutions for comparison
        u_T_exact = exact_T(t)
        u_p_exact = exact_p(t)
        
        # Monitor values at center
        T_center, p_center = probe_center(u)
        T_exact_center = (probe_T @ u_T_exact)[0]
        p_exact_center = (probe_p @ u_p_exact)[0]
        
        print(f't={t:4.2f}: T_center={T_center:6.3f} (err={T_center-T_exact_center:+7.4f}), '
              f'p_center={p_center:6.3f} (err={p_center-p_exact_center:+7.4f})')

        # Update visualization
        title.set_text(f'{field_name} at t = {t:.2f}')
        
        if args.field == 'temperature':
            new_data = u_T[basisT.nodal_dofs.flatten()]
        else:
            new_data = u_p[basisp.nodal_dofs.flatten()]
            
        field_plot.set_array(new_data)

    # Create and run animation
    animation = FuncAnimation(
        fig,
        update,
        evolve(0., u_init),
        repeat=False,
        interval=100,  # Slower for better visibility
    )
    
    if args.gif:
        gif_name = f'diffusion_{args.field}.gif'
        animation.save(Path(__file__).parent / gif_name, writer='pillow')
        print(f"Animation saved as {gif_name}")
    else:
        plt.show()
